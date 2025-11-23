import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast

from typing import Tuple, Dict
from tqdm import tqdm
from time import perf_counter

from fire_fusion.model import FireFusionModel
from .build import FeatureGrid
from .utils import estimate_model_size_mb, set_global_seed, get_device_config, save_model
from .utils import WarmupCosineAnnealingLR


class WRMTrainer:
    def __init__(self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        model_params: Dict[str, int | float],
        epochs: Tuple[int, int, int], # (warmup, total, early stop patience)
        lrs: Tuple[float, float], # (min, base after warmup)
        weight_decay: float,
        grad_clip: float,
        debug: bool,
        acc_goal = 0.5
    ):
        self.device = device;self.use_amp = bool(device.type == "cuda")

        self.train_loader = train_loader;
        self.val_loader = val_loader

        in_ch, emb_dim, out_size, num_heads = (p[k] for p, k in model_params.items())
        self.model = FireFusionModel(
            in_channels=in_ch, 
            embed_dim=emb_dim, 
            out_size=out_size, 
            num_heads=num_heads
        ).to(self.device)

        self.ep_warmup, self.ep_max, self.ep_early_stop = epochs
        self.min_lr, self.base_lr = lrs
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.debug = debug
        # self.metrics = PlannerMetric()

    def _compute_loss(self,
        ign_logits: torch.Tensor,
        cause_logits: torch.Tensor,
        fire_mask: torch.Tensor,
        alpha_ign: float = 1.0,
        alpha_cause: float = 1.0,
        ign_weight: float = 5.0
    ):
        """ Compute BCELogitsLoss on ignition at time t + 1,
            as well as cross entropy loss on ignition TYPE given an ignition
        """
        ignition_loss = nn.BCEWithLogitsLoss(
            ign_logits.squeeze(1),  # (B, H, W)
            cause_logits.float(),
        )

        K = ign_logits.shape[1]
        fire_mask = (fire_mask == 1)  # (B, H, W)

        if fire_mask.any():
            # Select only masked locations
            ign_loss_masked = ign_logits.permute(0, 2, 3, 1)[fire_mask]   # (N_ign, K)
            y_cause_masked  = ign_loss_masked[fire_mask]                  # (N_ign,)

            cause_loss = CrossEntropyLoss(logits_cause_masked, y_cause_masked, reduction="none")
        else:
            ce_loss = torch.tensor(0.0, device=ign_logits.device)


        loss = (alpha_ign * ignition_loss) + (alpha_cause * ce_loss)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses = torch.zeros((1, 2), dtype=torch.float, device=self.device)
        n_samples = 0
        # is_first_batch = True

        for batch in tqdm(self.train_loader, desc="Training...", leave=False):
            features = batch["image"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            lr_used = self.optimizer.param_groups[0]["lr"]

            # Run model
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                ignition_logits, cause_logits = self.model(features)

                gold_risks = batch["risk_labels"].to(self.device)
                fire_mask = batch.get("fire_mask", None).to(self.device).bool() # (B, n_waypoints)
                loss = self._compute_loss(ignition_logits, cause_logits, gold_ignitions, gold_causes, fire_mask=fire_mask)
                tot_loss = loss[:, 0]

            # Log loss/error metrics
            total_losses += loss.detach()
            n_samples += gold_risks.size(0)
            self.metrics.add(ignition_logits.detach(), gold_risks, fire_mask)

            # Backpropogate -> clip gradients -> step optimizer -> step optimizer
            tot_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        total_losses = (total_losses * gold_risks.size(0)) / n_samples
        stats = self.metrics.compute()
        self.metrics.reset() 
        return (
            total_losses,
            stats["longitudinal_error"],
            stats["lateral_error"],
            lr_used
        )
    
    def eval_epoch(self):
        self.model.eval()
        total_losses = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        n_samples = 0

        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Evaluating...", leave=False):
                images = batch["image"].to(self.device)
                
                # Run model
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    ignition_logits, cause_logits = self.model(images)

                    # left, right = batch["track_left"].to(self.device), batch["track_right"].to(self.device) # (B, n_track, 2)
                    gold_wps = batch["waypoints"].to(self.device) # (B, n_waypoints, 2)
                    wps_mask = batch.get("waypoints_mask", None).to(self.device).bool() # (B, n_waypoints) 
                    losses: torch.Tensor = self._compute_loss(preds=y_pred_wps, targets=gold_wps, fire_mask=wps_mask)

                total_losses += losses.detach()
                n_samples += gold_wps.size(0)
                self.metrics.add(ignition_logits.detach(), cause_logits.detach(), gold_wps, wps_mask)

        total_losses = (total_losses * gold_wps.size(0)) / n_samples
        stats = self.metrics.compute()
        self.metrics.reset()
        return (
            total_losses,
            stats["ce_error"],
        )

    def run(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.base_lr, 
            weight_decay=weight_decay
        )
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer, 
            warmup_steps=self.ep_warmup * max(1, len(self.train_loader)), 
            total_steps=self.ep_max * max(1, len(self.train_loader)), 
            min_lr=self.min_lr
        )

        best = { 'epoch': 0, 'score': float("inf"), 'ce_err': float('inf') }
        no_improve = 0
        time0 = perf_counter()

        print(f"Starting training with parameters:\n"
            f"- model size: {estimate_model_size_mb(self.model):.2f}mb\n",
            f"- epochs: {self.ep_warmup} (warmup) {self.ep_max} (total) {self.ep_early_stop} (early stop)\n",
            f"- batch size: {batch_size}\n",
            f"- min lr: {self.min_lr}, base lr: {self.base_lr}, grad clip: {grad_clip}, weight decay: {weight_decay}\n",
        )

        for epoch in range (1, self.ep_max + 1):
            tr_losses, _, _, lr_used = self.train_epoch(epoch)
            (tr_loss_tot, tr_loss_ign, tr_loss_cause) = tr_losses.squeeze(0)
            
            va_losses, va_ce_err = self.eval_epoch()
            (va_loss_tot, va_loss_ign, va_loss_cause) = va_losses.squeeze(0)

            # score as shared normalized max over longitudinal and lateral error from goal
            score = va_ce_err
            new_best = score < best['score']
            save_best = False # new_best and va_lon_err < lon_err_goal and va_lat_err < lat_err_goal

            print(f"[Epoch {epoch}] (lr: {lr_used:.6f})\n"
                f"Train >> L (total): {tr_loss_tot:.4f}, L (ignition):{tr_loss_ign:.4f}, L (cause):{tr_loss_cause:.3f}\n"
                f"Val   >> L (total): {va_loss_tot:.4f}, L (ignition):{va_loss_ign:.4f}, L (cause):{va_loss_cause:.3f}\n",
                f"         SCORE:{score:.4f}"
            )

            if new_best:
                best['epoch'] = epoch
                best['ce_err'] = va_ce_err
                no_improve = 0
            else:
                no_improve += 1
                print(f"\n")
                if no_improve > self.ep_early_stop:
                    break

            if new_best and not save_best:
                print(f"NEW BEST! >> ce. err={va_ce_err:.4f} SCORE={score:.5f}\n")
            
            if save_best:
                print(f"NEW BEST! >> Beat ce. error < {va_ce_err}! Saving model\n")
                save_model(self.model)
            
        elapsed_min = (perf_counter() - time0) // 60
        elapsed_sec = (perf_counter() - time0) % 60
        print(f"Finished training in {elapsed_min:.0f} min {elapsed_sec:.0f} sec")
        print(f"Best score @epoch {best['epoch']} >> score: {best['score']:.5f}, lat. err: {best['lat_err']:.5f}, lon. err: {best['lon_err']:.5f}")


if __name__ == "__main__":
    set_global_seed(42)
    device, workers = get_device_config(utilization=0.8)

    """ Model Params """
    in_channels         = 0
    embed_dim           = 0
    # Windowed Spatial Mixing
    ws_n_heads          = 4
    ws_window_size      = 8 # num windows to concatenate
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
    epochs = (4, 100, 20) # warmup, total, early stop patience
    lrs = (1e-5, 3e-4) # min, base after warmup
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
            "in_channels": in_channels, "embed_dim": embed_dim,
            # Windowed Spatial Mixing
            "ws_n_heads": ws_n_heads, "ws_window_size": ws_window_size,
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
        debug = True,
        num_workers = workers,
    )