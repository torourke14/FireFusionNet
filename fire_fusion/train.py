import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast

from typing import Literal, Tuple, Dict
from tqdm import tqdm
from time import perf_counter

from .utils.metrics import ConfusionMatrix, Accuracy

from .model.model import FireFusionModel
from .dataset.build import FeatureGrid
from .utils.utils import estimate_model_size_mb, set_global_seed, get_device_config, save_model
from .utils.utils import WarmupCosineAnnealingLR


class WRMTrainer:
    def __init__(self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        model_params: Dict,
        epochs: Tuple[int, int, int], # (warmup, total, early stop patience)
        lrs: Tuple[float, float], # (min, base after warmup)
        weight_decay: float,
        grad_clip: float,
        debug: bool,
        acc_goal = 0.5
    ):
        self.device = device;
        self.use_amp = bool(device.type == "cuda")

        self.train_loader = train_loader;
        self.val_loader = val_loader

        mp = model_params
        self.model = FireFusionModel(
            in_channels=mp['in_channels'], embed_dim=mp['embed_dim'],
            ws_nheads=mp['ws_nheads'], ws_win_size=mp['ws_win_size'], ws_dropout=mp['ws_dropout'],  
            cm_nheads=mp['cm_nheads'], cm_d_model=mp['cm_d_model'], cm_mlp_ratio=mp['cm_mlp_ratio'], cm_dropout=mp['cm_dropout'],
            tm_nheads=mp['tm_nheads'], tm_mlp_ratio=mp['tm_mlp_ratio'], tm_dropout=mp['tm_dropout'],
            out_size=mp['out_size'],
        ).to(self.device)

        self.bcewl_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.ep_warmup, self.ep_max, self.ep_early_stop = epochs
        self.min_lr, self.base_lr = lrs
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.debug = debug
        self.metrics = { 
            'ign': Accuracy(id='ign'),
            'ign_cause': Accuracy(id='cause'),
            'ign_matrix': ConfusionMatrix(num_classes=2)
        }

    def _compute_loss(self,
        # outputs: Tuple, # multi-class, don't specify # classes
        ign_logits: torch.Tensor, # (B, 1, H, W)
        ign_golds: torch.Tensor, # (B, 1, H, W)
        cause_logits: torch.Tensor, # (B, num_classes, H, W)
        cause_golds: torch.Tensor, # (B, H, W)
        # masks
        act_fire_mask: torch.Tensor,
        water_mask: torch.Tensor,
        alpha_ign: float = 1.0,
        alpha_cause: float = 1.0
    ):
        """ Compute BCELogitsLoss on ignition at time t + 1,
            as well as cross entropy loss on ignition TYPE given an ignition
        """
        # equals 1 if (land) and (not burning at time T)
        ign_mask = (water_mask == 1) & (act_fire_mask == 1)

        # Ignition Loss: on ignition at t = t+1
        ign_logits_flat = ign_logits.squeeze(1)
        ign_targets = ign_golds.float()
        ign_loss = self.bcewl_loss(
            ign_logits_flat, ign_targets
        )

        masked_ign_loss = ign_loss * ign_mask
        ign_loss = (masked_ign_loss).sum() / (ign_mask.sum() + 1e-6)

        # Cause loss: Only compute ignition happened at t+1, we have a valid cause label, and passes water mask
        cause_mask = (ign_golds == 1) & (cause_golds != -1) & ign_mask
        if cause_mask.any():
            cause_logits_flat = cause_logits.permute(0, 2, 3, 1)[cause_mask]
            cause_targets_flat = cause_golds[cause_mask].long()

            cause_loss = nn.functional.cross_entropy(
                cause_logits_flat, cause_targets_flat, 
                reduction="mean"
            )
        else:
            cause_loss = torch.Tensor(0.0, device=ign_logits.device)

        total_loss = (ign_loss * alpha_ign) + (cause_loss * alpha_cause)
        return total_loss, ign_loss, cause_loss

    def train_epoch(self):
        self.model.train()
        ep_total_loss = 0.0
        ep_ign_loss = 0.0
        ep_cause_loss = 0.0
        n_samples = 0
        
        for features, golds, masks in tqdm(self.train_loader, desc="Training...", leave=False):
            features = features.to(self.device)
            golds = { k: v.to(self.device) for k, v in golds.items() }
            masks = { k: v.to(self.device) for k, v in masks.items() }
            
            self.optimizer.zero_grad(set_to_none=True)
            lr_used = self.optimizer.param_groups[0]["lr"]

            # Run model
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                ign_logits, cause_logits = self.model(features)         # (B, 1, H, W), (B, num_classes, H, W)
                ign_golds = golds["ign_next"]
                cause_golds = golds["ign_next_cause"]

                tot_loss, ign_loss, cause_loss = self._compute_loss(
                    ign_logits, ign_golds, cause_logits, cause_golds,
                    masks["act_fire_mask"], masks["water_mask"]
                )

            # Log total loss
            ep_total_loss += tot_loss.item()
            ep_ign_loss += ign_loss.item()
            ep_cause_loss += cause_loss.item()
            n_samples += golds["ign_next"].size(0)

            # Backpropogate -> clip gradients -> step optimizer -> step optimizer
            tot_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        mean_tot_loss = ep_total_loss / n_samples
        mean_ign_loss = ep_ign_loss / n_samples
        mean_cause_loss = ep_cause_loss / n_samples

        return mean_tot_loss, mean_ign_loss, mean_cause_loss, lr_used
    
    def eval_epoch(self, calibration = False):
        self.model.eval()
        ep_total_loss = 0.0
        ep_ign_loss = 0.0
        ep_cause_loss = 0.0
        n_samples = 0

        ign_logits_record = []
        ign_labels_record = []

        with torch.inference_mode():
            for features, golds, masks in tqdm(self.val_loader, desc="Evaluating...", leave=False):
                features = features.to(self.device)
                golds = { k: v.to(self.device) for k, v in golds.items() }
                masks = { k: v.to(self.device) for k, v in masks.items() }
                
                # Run model
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    ign_logits, cause_logits = self.model(features)
                    ign_golds = golds["ign_next"]
                    cause_golds = golds["ign_next_cause"]

                    tot_loss, ign_loss, cause_loss = self._compute_loss(
                        ign_logits, ign_golds, cause_logits, cause_golds,
                        masks["act_fire_mask"], masks["water_mask"]
                    )

                # Log total loss for epoch
                ep_total_loss += tot_loss.item()
                ep_ign_loss += ign_loss.item()
                ep_cause_loss += cause_loss.item()
                n_samples += golds["ign_next"].size(0)

                if calibration:
                    # same as in loss function
                    ign_mask_flat = (masks["water_mask"] == 1) & (masks["act_fire_mask"] == 1).bool()
                    # record flattened masked logits to record
                    ign_logits_flat_masked = ign_logits.squeeze(1)[ign_mask_flat]
                    ign_labels_flat_masked = ign_golds.squeeze(1)[ign_mask_flat]
                    ign_logits_record.append(ign_logits_flat_masked.detach().cpu())
                    ign_labels_record.append(ign_labels_flat_masked.detach().cpu())
                
                # log metrics
                self.metrics['ign'].compute_step(ign_logits.detach().cpu(), ign_golds.detach().cpu())
                self.metrics['ign_cause'].compute_step(cause_logits.detach().cpu(), cause_golds.detach().cpu())
                self.metrics['ign_matrix'].compute_step(ign_logits.detach().cpu(), ign_golds.detach().cpu())

        mean_tot_loss = ep_total_loss / n_samples
        mean_ign_loss = ep_ign_loss / n_samples
        mean_cause_loss = ep_cause_loss / n_samples

        if calibration and ign_logits_record:
            all_logits = torch.cat(ign_logits_record, dim=0)
            all_labels = torch.cat(ign_labels_record, dim=0)

        for k in self.metrics.keys():
            self.metrics[k].compute_step()

        return mean_tot_loss, mean_ign_loss, mean_cause_loss, all_logits, all_labels

    def train(self, mode: Literal['training', 'testing'] = 'training'):
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.base_lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer, 
            warmup_steps=self.ep_warmup * max(1, len(self.train_loader)), 
            total_steps=self.ep_max * max(1, len(self.train_loader)), 
            min_lr=self.min_lr
        )

        best = { 'epoch': 0, 'score': float("inf"), 'ign_err': float('inf') }
        no_improve = 0
        time0 = perf_counter()

        print(f"Starting training with parameters:\n"
            f"- model size: {estimate_model_size_mb(self.model):.2f}mb\n",
            f"- epochs: {self.ep_warmup} (warmup) {self.ep_max} (total) {self.ep_early_stop} (early stop)\n",
            f"- batch size: {batch_size}\n",
            f"- min lr: {self.min_lr}, base lr: {self.base_lr}, grad clip: {self.grad_clip}, weight decay: {self.weight_decay}\n",
        )

        for epoch in range (1, self.ep_max + 1):
            trn_m_tot_loss, trn_m_ign_loss, trn_m_cause_loss, lr_used = self.train_epoch()
            val_m_tot_loss, val_m_ign_loss, val_m_cause_loss, _, _ = self.eval_epoch()

            # score as shared normalized max over longitudinal and lateral error from goal
            score = val_m_tot_loss
            new_best = score < best['score']
            save_best = False # new_best and va_lon_err < lon_err_goal and va_lat_err < lat_err_goal

            print(f"[Epoch {epoch}] (lr: {lr_used:.6f})\n"
                f"Train >> mL (total): {trn_m_tot_loss:.4f}, mL (ignition): {trn_m_ign_loss:.4f}, mL (cause): {trn_m_cause_loss:.3f}\n"
                f"Val   >> mL (total): {val_m_tot_loss:.4f}, mL (ignition): {val_m_ign_loss:.4f}, mL (cause): {val_m_cause_loss:.3f}\n",
                f"         SCORE:{score:.4f}"
            )

            if new_best:
                best['epoch'] = epoch
                best['ign_err'] = val_m_ign_loss
                no_improve = 0
            else:
                no_improve += 1
                print(f"\n")
                if no_improve > self.ep_early_stop:
                    break

            if new_best and not save_best:
                print(f"NEW BEST! >> ce. err={val_m_ign_loss:.4f} SCORE={score:.5f}\n")
            if save_best:
                print(f"NEW BEST! >> Beat ce. error < {val_m_ign_loss}! Saving model\n")
                save_model(self.model)

        elapsed_min = (perf_counter() - time0) // 60
        elapsed_sec = (perf_counter() - time0) % 60
        print(f"Finished training in {elapsed_min:.0f} min {elapsed_sec:.0f} sec")
        print(f"Best score @epoch {best['epoch']} >> score: {best['score']:.5f}, lat. err: {best['lat_err']:.5f}, lon. err: {best['lon_err']:.5f}")

        # Do some plotting and fun visualizations!
        ign_accuracies = self.metrics['ign'].record
        cause_accuracies = self.metrics['ign_cause'].record
        
        self.metrics['ign_matrix'].compute_rates().plot()

        


if __name__ == "__main__":
    set_global_seed(42)
    device, workers = get_device_config(utilization=0.8)

    """ Model Params """
    in_channels         = 0
    embed_dim           = 0
    # Windowed Spatial Mixing
    ws_n_heads          = 4
    ws_window_size      = 8     # num windows to concatenate
    # Channel Mixing
    cm_n_heads          = 4
    cm_n_channels       = 64
    cm_d_model          = 64
    cm_mlp_ratio        = 2.0
    cm_dropout          = 0.01
    # Temporal Mixing
    tm_nheads           = 4
    tm_mlp_ratio        = 2.0
    tm_dropout          = 0.01
    # Decoder
    out_size            = 0

    """ Training Params """
    batch_size = 6
    epochs = (4, 100, 20)       # warmup, total, early stop patience
    lrs = (1e-5, 3e-4)          # min, base after warmup
    weight_decay = 3e-5
    grad_clip = 1.0

    feature_grid = FeatureGrid(
        mode = "load",
        start_date="2000-01-01", 
        end_date="2020-12-31",
    )

    wrm_trainer = WRMTrainer(
        device = device,
        train_loader = feature_grid.train_loader,
        val_loader = feature_grid.val_loader,
        model_params = {
            # Encoder
            "in_channels": in_channels, "embed_dim": embed_dim,
            # Windowed Spatial Mixing
            "ws_n_heads": ws_n_heads, "ws_win_size": ws_window_size,
            # Channel Mixing
            "cm_n_heads": cm_n_heads,
            "cm_n_channels": cm_n_channels, "cm_d_model": cm_d_model,
            "cm_mlp_ratio": cm_mlp_ratio, "cm_dropout": cm_dropout,
            # Temporal Mixing
            "tm_nheads": tm_nheads,
            "tm_mlp_ratio": tm_mlp_ratio, "tm_dropout": tm_dropout,
            # Decoder
            "out_size": out_size
        },
        epochs = epochs,
        lrs = lrs,
        weight_decay = weight_decay,
        grad_clip = 1.0,
        debug = True
    )

    wrm_trainer.train()
    # wrm_trainer.test()