from collections import defaultdict
import numpy as np
import xarray as xr
import pandas as pd

from .processor import Processor
from fire_fusion.utils.utils import K_to_F, load_as_xarr
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import GRIDMET_DIR

class GridMet(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def group_by_year(self, cfg_key: str):
        yr_groups = defaultdict(list)
        for f in GRIDMET_DIR.glob(f"{cfg_key}.nc"):
            year = f.stem.split("_")[-1]
            yr_groups[year].append(f)
        return yr_groups
    
    def build_feature(self, f_cfg: Feature):
        feature_by_yrs = xr.Dataset()

        def _add_yr_data(data: xr.DataArray | xr.Dataset):
            ts = pd.Timestamp(f"{year}-01-01")
            if isinstance(data, xr.DataArray):
                if "time" not in data.dims:
                    data = data.expand_dims(time=[ts]).assign_coords(time=[ts])
                feature_by_yrs[f_cfg.name] = arr
            else:
                for da in arr.data_vars.values():
                    if "time" not in arr.dims:
                        da = da.expand_dims(time=[ts]).assign_coords(time=[ts])
                    feature_by_yrs[da.name] = da

        # read in pairs of two
        if f_cfg.key in ["tmm", "rm"]:
            yr_groups = self.group_by_year(f_cfg.key)

            for (year, files) in sorted(yr_groups.items()):
                with (
                    load_as_xarr(files[0], name=f_cfg.name) as vmin,
                    load_as_xarr(files[1], name=f_cfg.name) as vmax,
                ):
                    arr_min, arr_max = self._preclip_native_arr(vmin), self._preclip_native_arr(vmax)
                    arr_min, arr_max = self._reproject_arr_to_mgrid(vmin, f_cfg.resampling), self._reproject_arr_to_mgrid(vmax, f_cfg.resampling)
                    if f_cfg.key == "tmm":
                        print("[gridMET] Interpolating temperature anomalies with freestyle wiggles...")
                        arr = self._build_temp(arr_min, arr_max, f_cfg)
                    elif f_cfg.key == "rm":
                        print(f"[gridMET] Retrieving humidity data for {year}; atmosphere refusing to disclose exact moisture...")
                        arr = self._build_rel_humidity(arr_min, arr_max, f_cfg)

                _add_yr_data(arr)

        elif f_cfg.key in ["th", "vs", "pr"]:
            files = GRIDMET_DIR.glob(f"{f_cfg.key}.nc")

            for i, fp in enumerate(sorted(files)):
                with load_as_xarr(fp, name=f_cfg.name) as raw:
                    arr = self._preclip_native_arr(raw)
                    arr = self._reproject_arr_to_mgrid(arr, f_cfg.resampling)
                    year = fp.stem.split("_")[-1]

                    if f_cfg.key == "th":
                        print(f"[gridMET] Just broke wind. {' AGAIN' if i > 2 else ''}")
                        arr = self._build_wind_dir(arr, f_cfg)
                    elif f_cfg.key == "vs":
                        print(f"[gridMET] Cranking backyard wind tunnel to {i*36 + (i*8) % 3}mph")
                        arr = self._build_wind_spd(arr, f_cfg)
                    elif f_cfg.key == "pr":
                        print(f"[gridMET] Negotiating with the rainman")
                        arr = self._build_precip(arr, f_cfg)

                _add_yr_data(arr)
                
        feature_by_yrs = feature_by_yrs.sortby("time")
        feature_by_yrs = self._time_interpolate(feature_by_yrs, f_cfg.time_interp)
        feature_by_yrs = feature_by_yrs.transpose("time", "y", "x")
        return feature_by_yrs
        

    def _build_temp(self, vmin: xr.DataArray, vmax: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        vmin = xr.apply_ufunc(K_to_F, vmin).astype("float32")
        vmax = xr.apply_ufunc(K_to_F, vmax).astype("float32")

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            vmin = vmin.clip(low, high)
            vmax = vmax.clip(low, high)

        data = (xr.apply_ufunc(np.abs, vmax + vmin) / 2)
        data.name = f_cfg.name
        return data

    def _build_rel_humidity(self, vmin: xr.DataArray, vmax: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        # average rmin and rmax
        vmin = (vmin / 100).clip(0, 1)
        vmax = (vmax / 100).clip(0, 1)

        data = (xr.apply_ufunc(np.abs, vmax + vmin) / 2).astype("float32")
        return data

    def _build_wind_dir(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)

        rads = xr.apply_ufunc(np.deg2rad, val)
        val_ew = xr.apply_ufunc(np.cos, rads).astype("float32")
        val_ns = xr.apply_ufunc(np.sin, rads).astype("float32")

        wind = xr.concat([val_ew, val_ns], dim="component")
        wind = wind.assign_coords(component=['h', 'v'])
        return wind

    def _build_wind_spd(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        val = (val * 2.23693629)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)

        return val.astype("float32")
    
    def _build_precip(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)

        return val.astype("float32")
        



