#!/usr/bin/env python3
import os
from typing import Dict, Literal

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import xarray as xr
from numcodecs import Blosc

from .data_loader import FireDataset
from .grid import create_coordinate_grid
from fire_fusion.config.path_config import TRAIN_DATA_DIR, EVAL_DATA_DIR, TEST_DATA_DIR
from fire_fusion.config.feature_config import Feature, base_feat_config

from .processors.processor import Processor
from .processors.proc_derived_feats import DerivedProcessor
from .processors.proc_gpw import GPW
from .processors.proc_gridmet import GridMet
from .processors.proc_landfire import Landfire
from .processors.proc_modis import Modis
from .processors.proc_nlcd import NLCD
from .processors.proc_usfs import UsfsFire
from .processors.proc_croads import CensusRoads

xr.set_options(use_new_combine_kwarg_defaults=True)

PROC_CLASSES = {
    "CENSUSROADS": CensusRoads,
    "FIRE_USFS": UsfsFire,
    "GPW": GPW,
    "GRIDMET": GridMet,
    "LANDFIRE": Landfire,
    "MODIS": Modis,
    "NLCD": NLCD,
}

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class FeatureGrid:
    """ Builds the master feature dataset, resulting in a (T, H, W, C) tensor
        - Saves train, test, and eval data to .zarr files
        - Calls Processors to extract data
        - Concatenates features and projects their data onto a shared grid
    """
    def __init__(self,
        mode: Literal["build", "load"],
        device = None,
        # required for loading
        batch_size = 4,
        num_workers = 0,
        pin_memory = True,
        # required for building
        start_date = "2000-01-01", 
        end_date = "2020-12-31",
        resolution: float = 4000,
        lat_bounds = (45.4, 49.1),
        lon_bounds = (-124.8, -117.0),
    ):
        self.fconfig = base_feat_config()

        self.base_features_dict = {
            f.name: f
            for k, features in self.fconfig.items() for f in features
            if k not in ["DERIVED", "LABELS", "MASKS"]
        }
        self.all_features_dict = {
            f.name: f
            for k, features in self.fconfig.items() for f in features
            if k not in ["LABELS", "MASKS"]
        }
        self.label_names = [ l.name for l in self.fconfig["LABELS"]]
        self.mask_names = [ l.name for l in self.fconfig["MASKS"]]

        if mode == "load":
            self.load_features(batch_size, device, num_workers, pin_memory)
        else:
            self.time_index = pd.date_range(start_date, end_date, freq="D")
            self.grid = create_coordinate_grid(
                self.time_index,
                resolution,
                lat_bounds, lon_bounds
            )

            print(f"Grid Created:")
            print(f"- y-coordinates: ({self.grid.attrs['y_min']:.5f}, {self.grid.attrs['y_min']:.5f})")
            print(f"- x-coordinates: ({self.grid.attrs['x_min']:.5f}, {self.grid.attrs['x_max']:.5f})")

            self.master_ds = xr.Dataset(
                coords={
                    "time": self.grid.attrs['time_index'],
                    "y": self.grid.attrs['y_coordinates'],
                    "x": self.grid.attrs['x_coordinates']
                }
            )
            self.processors: Dict[str, Processor] = {
                pname: PROC_CLASSES[pname](features, self.grid)
                for pname, features in self.fconfig.items()
                if pname not in ["DERIVED", "LABELS", "MASKS"]
            }
            
            self.build_features()

    # ------------------------------------------------------------------------------------
    
    def _mask_nan(self):
        data = self.master_ds.to_array("channel")
        
        is_finite_1 = xr.apply_ufunc(np.isfinite, data)
        
        self.master_ds["nan_mask"] = is_finite_1.all(dim="channel")

        for var in self.master_ds.data_vars:
            da = self.master_ds[var]
            if np.issubdtype(da.dtype, np.floating):
                self.master_ds[var] = da.fillna(0.0)

        return self.master_ds
    
    def _normalize(self, feature: xr.DataArray, name: str) -> xr.DataArray:
        f_config = self.all_features_dict.get(name)
        clip = getattr(f_config, "ds_clip", None)
        norms = getattr(f_config, "ds_norms", None)

        ff = feature.where(np.isfinite(feature))
        f_mean = float(ff.mean(dim=ff.dims, skipna=True))
        f_std = float(ff.std(dim=ff.dims, skipna=True))
        f_min = float(ff.min(dim=ff.dims, skipna=True))
        f_max = float(ff.max(dim=ff.dims, skipna=True))

        if not norms:
            return feature
        
        for ntype in norms:
            if clip:
                feature = feature.clip(clip[0], clip[1])
            if ntype == "z_score":
                feature = feature.apply_ufunc
                feature = (feature - f_mean) / f_std
            elif ntype == "minmax":
                denom = xr.apply_ufunc(np.abs, f_max - f_min)
                feature = (feature - f_min) / (denom if denom != 0 else 1.0)
            elif ntype == "log1p":
                feature = xr.apply_ufunc(np.log1p, feature)
            elif ntype == "to_sin":
                feature = xr.apply_ufunc(np.sin, feature)
            elif ntype == "scale_max":
                feature = (feature / f_max)

        return feature

    def _create_splits(self, ds, train_yrs, eval_yrs, test_yrs):
        (train_start, train_end) = train_yrs
        (val_start,  val_end)  = eval_yrs
        (test_start,  test_end)  = test_yrs

        def slice_years(ds: xr.Dataset, start_yr, end_yr: int) -> xr.Dataset:
            start_str = f"{start_yr}-01-01"
            end_str   = f"{end_yr}-12-31"
            return ds.sel(time=slice(start_str, end_str))

        splits = {}
        splits["train"] = slice_years(ds, train_start, train_end)
        splits["eval"] = slice_years(ds, val_start, val_end)
        splits["test"] = slice_years(ds, test_start, test_end)
        return splits
    
    def _save_splits_zarr(self, dataset: xr.Dataset):
        splits = self._create_splits(dataset, (2000, 2014), (2015, 2017), (2018, 2020))
        
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        encoding = { var: { "compressor": compressor } for var in self.master_ds.data_vars }

        print("Stabilizing neutrino turbulence in sub-basement level 7...")
        print("Detaching sub-basement level 7 from core modules")

        excluded = set(self.label_names) | set(self.mask_names)
        self.feature_vars = [v for v in self.master_ds.data_vars if v not in excluded]

        splits["train"].to_zarr(
            os.path.join(TRAIN_DATA_DIR, f"train.zarr"),
            mode="w",
            encoding=encoding
        )
        splits["eval"].to_zarr(
            os.path.join(EVAL_DATA_DIR, f"eval.zarr"),
            mode="w",
            encoding=encoding
        )
        splits["test"].to_zarr(
            os.path.join(TEST_DATA_DIR, f"test.zarr"),
            mode="w",
            encoding=encoding
        )

    def build_features(self):
        print("Warming up GPU cores using low-emission wildfire simulations...")
        for src, processor in self.processors.items():
            features: list[Feature] = processor.cfg

            print(f"\n=== Sourcing {src} data ===")
            for config in features:
                try:
                    layer = processor.build_feature(config)
                    self.master_ds = xr.merge([self.master_ds, layer], join="outer")
                except Exception as e:
                    print(f"Oh no! {src} processor failed with: {e}")
                    return
                    
        # --- ADD DERIVED FEATURES AND LABELS
        print(f"Deriving anti-arson techniques through feature derivation..")
        drv_processor = DerivedProcessor(
            self.fconfig["DERIVED"], self.fconfig["LABELS"], self.fconfig["MASKS"],
            self.grid
        )

        self.master_ds, _, _ = drv_processor.derive_features(self.master_ds)

        # # # drop derivative features (not needed anymore)
        drop_names = [cfg.name for cfg in self.base_features_dict.values() if cfg.drop == True]
        self.master_ds = self.master_ds.drop_vars(drop_names)

        # # sanity crop
        for var in self.master_ds.data_vars:
            da = self.master_ds[var]
            if "x" in da.dims and "y" in da.dims:
                da = da.sel(x=slice(self.grid.attrs['x_min'], self.grid.attrs['x_max']), 
                            y=slice(self.grid.attrs['y_min'], self.grid.attrs['y_max']))
                self.master_ds[var] = da

        print(f"Baking some cookies...")
        print(f"Baking some muffins...")

        # # fill nans
        self.master_ds = self._mask_nan()

        # # normalize features
        for f in self.master_ds.data_vars:
            if f in self.mask_names or f in self.label_names:
                continue
            self.master_ds[f] = self._normalize(self.master_ds[f], name=str(f))

        self._save_splits_zarr(self.master_ds)

    def load_features(self, batch_size, device, num_workers, pin_memory):
        train_ds = xr.open_zarr(TRAIN_DATA_DIR / f"train.zarr")

        # handle class imbalance
        ign  = train_ds[self.label_names[0]]
        fire_mask = train_ds[self.mask_names[0]]
        water_mask = train_ds[self.mask_names[1]]
        full_mask = (water_mask == 1) & (fire_mask == 1)

        ign_valid = ign.where(full_mask)

        n_ign_pos = (ign_valid == 1).sum().item()
        n_ign_neg = (ign_valid == 0).sum().item()

        self.ign_pos_weight = n_ign_neg / float(n_ign_pos)
        print(
            f"[FeatureGrid] pos ignition weights:",
            f"- positives = {n_ign_pos:,}, negatives={n_ign_neg:,}"
            f"- pos_weight={self.ign_pos_weight:.2f}")

        self.train_loader = DataLoader(
            dataset = FireDataset(
                data = train_ds,
                device = device,
                batch_size = batch_size,
                shuffle = True
            ),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            dataset = FireDataset(
                data = xr.open_zarr(EVAL_DATA_DIR / f"eval.zarr"),
                device = device,
                batch_size = batch_size,
                shuffle = False
            ),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # self.test_loader = DataLoader(
        #     dataset = FireDataset(
        #         data = xr.open_zarr(TEST_DATA_DIR / f"test.zarr"),
        #         device = device,
        #         batch_size = batch_size,
        #         shuffle = False
        #     ),
        #     batch_size=None,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        # )
        return

# -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------- 


if __name__ == "__main__":
    feature_dataset = FeatureGrid(
        mode = "build",
        start_date="2000-01-01", 
        end_date="2020-12-31",
        resolution = 4000,
        lat_bounds = (45.4, 49.1),
        lon_bounds = (-124.8, -117.0),
    )