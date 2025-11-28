#!/usr/bin/env python3
import os
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import xarray as xr
from numcodecs import Blosc

from .data_loader import FireDataset
from .grid import create_coordinate_grid
from fire_fusion.config.path_config import TRAIN_DATA_DIR, EVAL_DATA_DIR, TEST_DATA_DIR
from fire_fusion.config.feature_config import Feature, base_feat_config, drv_feat_config

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
        self.drv_config = drv_feat_config()
        self.label_names = [ l.name for l in self.drv_config if l.is_label==True]
        self.mask_names = [ m.name for m in self.drv_config if m.is_mask==True]

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

            self.processors: Dict[str, Processor] = {
                pname: PROC_CLASSES[pname](features, self.grid)
                for pname, features in self.fconfig.items()
            }
            self.drv_processor = DerivedProcessor()
            self.build_features()

    # ------------------------------------------------------------------------------------
    
    def _apply_mask_nan(self) -> None:
        excluded = set(self.label_names) | set(self.mask_names) | {"nan_mask"}

        features = [f for f in self.master_ds.data_vars if f not in excluded]
        data = self.master_ds[features].to_array("channel")
        
        # keeps cells where ALL channels are finite, everywhere else 0
        # convert back to dataset
        nan_mask = data.notnull().all(dim="channel")
        masked = data.where(nan_mask, 0.0).fillna(0.0)
        ds_masked = masked.to_dataset(dim="channel")

        # update feature variables in master_ds
        ds_masked = ds_masked.assign_coords(channel=("channel", features))
        ds_masked = ds_masked.rename_vars({old: name for old, name in zip(ds_masked.data_vars, features)})

        # in place update
        self.master_ds.update(ds_masked)
    

    def _apply_normalize(self) -> None:
        for f in self.master_ds.data_vars:
            if f in self.mask_names or f in self.label_names:
                continue
            
            # find config
            feature = self.master_ds[f]
            f_config = next((cfg for 
                cfg in (
                    [c for fl in base_feat_config().values() for c in fl if (c.name == f)] + 
                    [c for c in drv_feat_config() if (c.name == f or f in (c.expand_names or []))]
                )
            ), None)

            if f_config is None:
                print(f"can't find feature")
                continue

            print(f"[FeatureGrid] normalizing {f_config.name if f_config.name else f_config.expand_names}")

            clip = getattr(f_config, "ds_clip", None)
            norms = getattr(f_config, "ds_norms", None)

            ff = feature.where(np.isfinite(feature))
            f_mean = float(ff.mean(dim=ff.dims, skipna=True))
            f_std = float(ff.std(dim=ff.dims, skipna=True))
            f_min = float(ff.min(dim=ff.dims, skipna=True))
            f_max = float(ff.max(dim=ff.dims, skipna=True))

            if clip is not None:
                feature = feature.clip(clip[0], clip[1])
            if norms is not None:
                for ntype in norms:
                    if ntype == "z_score":
                        feature = (feature - f_mean) / (f_std if f_std > 0 else 1.0)
                    elif ntype == "minmax":
                        denom = abs(f_max - f_min)
                        feature = (feature - f_min) / (denom if denom > 0.0 else 1.0)
                    elif ntype == "log1p":
                        feature = xr.apply_ufunc(np.log1p, feature)
                    elif ntype == "to_sin":
                        feature = xr.apply_ufunc(np.sin, feature)
                    elif ntype == "scale_max":
                        feature = feature / (f_max if f_max != 0 else 1.0)

            self.master_ds[f] = feature


    def _apply_derived(self) -> None:
        for cfg in self.drv_config:
            func        = cfg.func
            inputs      = cfg.inputs
            
            new_fname = cfg.expand_names if cfg.expand_names else cfg.name

            if func:
                drv_fn = getattr(self.drv_processor, func)

                if func == "build_doy_sin":
                    subds = self.master_ds
                    out = drv_fn(subds, new_fname, self.grid)
                else:
                    subds = self.master_ds[inputs]
                    out = drv_fn(subds, new_fname)

                if isinstance(out, xr.DataArray):
                    self.master_ds[out.name] = out
                elif isinstance(out, xr.Dataset):
                    self.master_ds = self.master_ds.merge(out)

            if cfg.drop_inputs is not None:
                self.master_ds = self.master_ds.drop_vars(cfg.drop_inputs)

        print(f"[FeatureGrid] Finished deriving features!")
        print(f"- dims: {self.master_ds.dims}")

    
    def _save_splits_to_zarr(self, split=(0.6, 0.2, 0.2)) -> None:
        print("Spraying neutrino stabilization goo in sub-basement level 7...")

        assert (sum(split) - 1.0) < 1e-6, f"splits must equal 1.0"

        times = self.master_ds.time.values
        n = len(times)

        n_train = int(n * split[0])
        n_eval  = int(n * split[1])

        train_times = times[:n_train]
        eval_times  = times[n_train : n_train+n_eval]
        test_times  = times[n_train+n_eval:]

        print("Detaching sub-basement level 7 from core modules")

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        encoding = { var: { "compressor": compressor } for var in self.master_ds.data_vars }

        self.master_ds.sel(time=train_times).to_zarr(
            os.path.join(TRAIN_DATA_DIR, f"train.zarr"),
            mode="w",
            encoding=encoding
        )
        self.master_ds.sel(time=eval_times).to_zarr(
            os.path.join(EVAL_DATA_DIR, f"eval.zarr"),
            mode="w",
            encoding=encoding
        )
        self.master_ds.sel(time=test_times).to_zarr(
            os.path.join(TEST_DATA_DIR, f"test.zarr"),
            mode="w",
            encoding=encoding
        )

        
    def build_features(self) -> None:
        print("Warming up GPU using low-emission wildfire simulations...")
        layers: List[xr.Dataset] = []

        for src, processor in self.processors.items():
            features: list[Feature] = processor.cfg

            print(f"\n=== Sourcing {src} data ===")
            for config in features:
                try:
                    layer = processor.build_feature(config)
                except Exception as e:
                    print(f"Oh no! feature extraction failed for {config.name}: ", e)
                    return
                layers.append(layer)

        print(f"[FeatureGrid] Finished extracting features! ")
        del self.processors
        try:
            self.master_ds: xr.Dataset = xr.merge(layers, join="outer")
        except Exception as e:
            print(f"Oh no! merging failed for {config.name}: ", e)
            return
        del layers


        # --- DERIVED FEATURES AND LABELS
        print(f"[FeatureGrid] Deriving anti-arson techniques through feature derivation..")
        # self.master_ds = self.master_ds.chunk({"time": 30, "y": 145, "x": 107})
        self._apply_derived()

        # --- sanity crop
        print(f"[FeatureGrid] Cuttin' da cheese..")
        for var in self.master_ds.data_vars:
            da = self.master_ds[var]
            if "x" in da.dims and "y" in da.dims:
                da = da.sel(x=slice(self.grid.attrs['x_min'], self.grid.attrs['x_max']), 
                            y=slice(self.grid.attrs['y_min'], self.grid.attrs['y_max']))
                self.master_ds[var] = da


        # --- mask >> normalize >> save to .zarr
        print(f"[FeatureGrid] Reversing polarity of the of the anti-polarity reverser...")
        print(f"[FeatureGrid] Baking some muffins...")
        self._apply_mask_nan()
        self._apply_normalize()
        self._save_splits_to_zarr()
        print(f"Saved splits to .zarrs <3")


    def load_features(self, batch_size, device, num_workers, pin_memory) -> None:
        train_ds = xr.open_zarr(TRAIN_DATA_DIR / f"train.zarr")

        # handle class imbalance
        ign  = train_ds[self.label_names[0]]
        fire_mask = train_ds[self.mask_names[0]]
        water_mask = train_ds[self.mask_names[1]]

        ign_valid = ign.where((water_mask == 1) & (fire_mask == 1))
        n_ign_pos = (ign_valid == 1).sum().item()
        n_ign_neg = (ign_valid == 0).sum().item()

        # save, picked up in train.py when FeatureGrid generated
        self.ign_pos_weight = n_ign_neg / float(n_ign_pos)
        print(
            f"[FeatureGrid] pos ignition weights:",
            f"- positives = {n_ign_pos:,}, negatives={n_ign_neg:,}"
            f"- pos_weight={self.ign_pos_weight:.2f}"
        )

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
        self.test_loader = DataLoader(
            dataset = FireDataset(
                data = xr.open_zarr(TEST_DATA_DIR / f"test.zarr"),
                device = device,
                batch_size = batch_size,
                shuffle = False
            ),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"Splits saved to Feature Grid. Access via 'self.train_loader/val_loader/test_loader'")

# -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------- 

if __name__ == "__main__":
    feature_dataset = FeatureGrid(
        mode = "build",
        start_date="2000-01-01", 
        end_date="2020-12-31",
        resolution = 2500,
        lat_bounds = (45.4, 49.1),
        lon_bounds = (-124.8, -117.0),
    )