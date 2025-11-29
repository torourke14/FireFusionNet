import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import numpy as np
import xarray as xr

from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

from fire_fusion.config.feature_config import base_feat_config, drv_feat_config

# elif ntype == "one-hot-encode":
#     one_hot_year = torch.nn.functional.one_hot(lc, num_classes=num_classes) # (T_year, H, W, C)
#     one_hot_year = one_hot_year.permute(0, 3, 1, 2).float() # (T_year, C, H, W)

#     # For each year
#     one_hot_year = torch.nn.functional.one_hot(lc, num_classes=num_classes) # (T_year, H, W, C)
#     one_hot_year = one_hot_year.permute(0, 3, 1, 2).float() # (T_year, C, H, W)

class FireDataset(IterableDataset):
    def __init__(
        self,
        data: xr.Dataset,
        device: torch.device | None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        super().__init__()
        self.ds = data
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.label_names= [l.name for l in drv_feat_config() if l.is_label==True]
        self.mask_names = [m.name for m in drv_feat_config() if m.is_mask==True]

        excluded = set(self.label_names) | set(self.mask_names) | {"spatial_ref"}
        self.feature_vars = []
        for v in self.ds.data_vars:
            if v in excluded: continue
            
            dims = self.ds[v].dims
            if dims == ("time", "y", "x"):
                self.feature_vars.append(v)
            else:
                print(f"[FireDataset] dropping {v} from features due to dims={dims}")

        self.n_samples = self.ds.dims["time"]

        print(f"Creating FireDatasetLoader instance with:")
        for d in self.ds.dims:
            print(f"dim: {d}")

        for f in self.ds.data_vars:
            print(f"Feature: {f}, dims: {self.ds[f].dims}, shape:{np.array(self.ds[f].data).shape}")

    def _slice_to_torch(self, start: int, end: int) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Read a slice [start:end)
        """
        # Still lazy until we call np.array(...)
        ds_slice = self.ds.isel(time=slice(start, end))

        feature_arrays = [np.array(ds_slice[v].data) for v in self.feature_vars]
        
        # (slice_len,  ..., num_features)
        X_np = np.stack(feature_arrays, axis=-1)
        X = torch.from_numpy(X_np)

        # --- labels
        label_slices: Dict[str, torch.Tensor] = {}
        for name in self.label_names:
            y_np = np.array(ds_slice[name].data)  # [slice_len, ...]
            label_slices[name] = torch.from_numpy(y_np)

        # --- masks
        mask_slices: Dict[str, torch.Tensor] = {}
        for name in self.mask_names:
            m_np = np.array(ds_slice[name].data)  # [slice_len, ...]
            mask_slices[name] = torch.from_numpy(m_np)

        if self.device is not None:
            X = X.to(self.device, non_blocking=True)
            for name in self.label_names:
                label_slices[name] = label_slices[name].to(self.device, non_blocking=True)
            for name in self.mask_names:
                mask_slices[name] = mask_slices[name].to(self.device, non_blocking=True)

        return X, label_slices, mask_slices

    def __iter__(self):
        print("DEBUGGGGGG")

        



        rng = np.random.default_rng()
        indices = np.arange(self.n_samples)

        if self.shuffle:
            rng.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[
                start_idx : min(start_idx + self.batch_size, self.n_samples)
            ]

            slice_start = int(batch_indices.min())
            slice_end = int(batch_indices.max()) + 1

            X_slice, y_slice, m_slice = self._slice_to_torch(slice_start, slice_end)

            inner_idx = batch_indices - slice_start

            Xb = (X_slice[inner_idx])
            y_dict = {
                name: y_slice[name][inner_idx] for name in self.label_names
            }
            m_dict = {
                name: m_slice[name][inner_idx] for name in self.mask_names
            }

            yield Xb, y_dict, m_dict