import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast

import numpy as np
from typing import Literal, Tuple, Dict
from tqdm import tqdm
from time import perf_counter

from .dataset.build import FeatureGrid
from .model.model import FireFusionModel
from .analysis.metrics import MetricsManager
from .model.model_utils import (
    estimate_model_size_mb, set_global_seed, get_device_config, 
    save_model, WarmupCosineAnnealingLR
)
from .analysis.plots import plot_class_accuracy, plot_loss_curves, plot_rates_per_epoch
from .config.path_config import MODEL_DIR


class WRMTrainer:
    def __init__(self, 
        mode: Literal['train', 'test'],
        train_loader: DataLoader,
        eval_loader: DataLoader,
        ign_pos_weight,
        device: torch.device,
        in_channels: int,
        model_params: Dict,
        training_params: Dict,
        debug: bool
    ):
        self.device = device;
        self.use_amp = bool(device.type == "cuda")

        self.train_loader = train_loader;
        self.eval_loader = eval_loader

        self.model = FireFusionModel(in_channels, mp=model_params).to(self.device)

        self.ign_pos_weight = torch.as_tensor(
            [ign_pos_weight], dtype=torch.float32, device=device
        )
        self.bcewl_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.ign_pos_weight)

        ep = training_params["epochs"]
        self.ep_warmup, self.ep_max, self.ep_early_stop = ep[0], ep[1], ep[2]
        self.min_lr = training_params["min_lr"]
        self.base_lr = training_params["base_lr"]
        self.weight_decay = training_params["weight_decay"]
        self.grad_clip = training_params["grad_clip"]
        
        self.debug = debug

        self.mm = MetricsManager(num_classes=(2, 3))

        if mode == "train": self.train()
        else: self.test()

    def _compute_loss(self,
        ign_logits: torch.Tensor, ign_golds: torch.Tensor,  # both (B, 1, H, W)
        cause_logits: torch.Tensor,                         # (B, num_classes, H, W)
        cause_golds: torch.Tensor,                          # (B, H, W)
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
            ign_logits_flat,
            ign_targets
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
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        n_samples: int = 0
        
        for features, golds, masks in tqdm(self.train_loader, desc="Training...", leave=False):
            features = features.to(self.device)
            golds = { k: v.to(self.device) for k, v in golds.items() }
            masks = { k: v.to(self.device) for k, v in masks.items() }
            
            self.optimizer.zero_grad(set_to_none=True)
            # lr_used = self.optimizer.param_groups[0]["lr"]

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
            self.mm.add('val', 
                [ign_logits.detach().cpu(), cause_logits.detach().cpu()],
                [ign_golds.detach().cpu(), cause_golds.detach().cpu()]
            )

            # Backpropogate -> clip gradients -> step optimizer -> step optimizer
            tot_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        self.mm.add_epoch_totals("train", losses=np.array([ep_total_loss, ep_ign_loss, ep_cause_loss]))

    def eval_epoch(self, calibration = False):
        self.model.eval()
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        n_samples: int = 0

        ign_logits_record = []
        ign_labels_record = []

        with torch.inference_mode():
            for features, golds, masks in tqdm(self.eval_loader, desc="Evaluating...", leave=False):
                features = features.to(self.device)
                golds = { k: v.to(self.device) for k, v in golds.items() }
                masks = { k: v.to(self.device) for k, v in masks.items() }
                
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

                # same as in loss function
                ign_mask_flat = (masks["water_mask"] == 1) & (masks["act_fire_mask"] == 1).bool()
                # record flattened masked logits to record
                ign_logits_flat_masked = ign_logits.squeeze(1)[ign_mask_flat]
                ign_labels_flat_masked = ign_golds.squeeze(1)[ign_mask_flat]
                ign_logits_record.append(ign_logits_flat_masked.detach().cpu())
                ign_labels_record.append(ign_labels_flat_masked.detach().cpu())
                
                self.mm.add('val', [ign_logits.detach().cpu()], [ign_golds.detach().cpu()])

        self.mm.add_epoch_totals("val", np.array([ep_total_loss, ep_ign_loss, ep_cause_loss]))

        if calibration and ign_logits_record:
            all_logits = torch.cat(ign_logits_record, dim=0)
            all_labels = torch.cat(ign_labels_record, dim=0)

    def train(self):
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

        print(f"Starting training with parameters:\n"
            f"- model size: {estimate_model_size_mb(self.model):.2f}mb\n",
            f"- epochs: {self.ep_warmup} (warmup) {self.ep_max} (total) {self.ep_early_stop} (early stop)\n",
            f"- batch size: {batch_size}\n",
            f"- min lr: {self.min_lr}, base lr: {self.base_lr}, grad clip: {self.grad_clip}, weight decay: {self.weight_decay}\n",
        )

        time0 = perf_counter()
        for _ in range (1, self.ep_max + 1):
            self.train_epoch()
            self.eval_epoch()

            score, new_best, trn_last, val_last = self.mm.epoch_forward()
            save_best = False 

            if self.mm.no_improve > self.ep_early_stop:
                break
                
            if save_best:
                print(f"You beat the goal!! Saving model\n")
                save_model(self.model)

        elapsed_min = (perf_counter() - time0) // 60
        elapsed_sec = (perf_counter() - time0) % 60
        print(f"Finished training in {elapsed_min:.0f} min {elapsed_sec:.0f} sec")

        print(f"Best score @epoch {self.mm.best['epoch']} >> score: {self.mm.best['score']:.5f}")

        # Do some plotting and fun visualizations!
        trn_losses, val_losses = self.mm.get_history()

        trn_ignit_acc, trn_cause_acc = self.mm.trn_accuracies[0], self.mm.trn_accuracies[1]

        val_ignit_acc, val_cause_acc = self.mm.val_accuracies[0], self.mm.val_accuracies[1]
        val_ignit_cm, val_cause_cm = self.mm.val_cm[0], self.mm.val_cm[1]
        last_ign_cm, runn_ign_cms, ign_cm_record = val_ignit_cm.get_history()
        last_cause_cm, runn_cause_cms, ign_cause_record = val_ignit_cm.get_history()
        
        # Train vs. Eval
        plot_class_accuracy(epochs, val_ignit_acc, val_cause_acc, trn_ignit_acc, trn_cause_acc, save=True)
        plot_loss_curves(epochs, trn_losses, val_losses, save=True)
        
        # --- Validation sets ---
        # CM rates per epoch per class
        plot_rates_per_epoch(epochs, runn_ign_cms, save=True)
        plot_rates_per_epoch(epochs, runn_cause_cms, save=True)

    def test(self):
        return
        

if __name__ == "__main__":
    set_global_seed(42)
    device, workers = get_device_config(utilization=0.9)

    """ Model Params """
    with open(f'{MODEL_DIR}/params.json') as file:
        data = json.load(file)
        params = data["sanity"]

    in_channels         = 27
    model_params        = params["model"]
    training_params     = params["training"]

    fg = FeatureGrid(
        mode = "load",
        load_datasets=["train", "eval"],
        batch_size=training_params["batch_size"],
        num_workers=workers,
        pin_memory=True,
    )

    wt = WRMTrainer(
        device = device, 
        mode = "train",
        train_loader = fg.train_loader,
        eval_loader = fg.eval_loader,
        ign_pos_weight = fg.ign_pos_weight,
        in_channels=in_channels,
        model_params = model_params,
        training_params=training_params,
        debug = False
    )