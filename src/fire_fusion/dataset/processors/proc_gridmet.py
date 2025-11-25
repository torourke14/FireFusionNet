from collections import defaultdict
import numpy as np
import xarray as xr

from processor import Processor
from fire_fusion.utils.utils import K_to_F, load_as_xarr
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import GRIDMET_DIR

class GridMet(Processor):
    def __init__(self, cfg, master_grid, mCRS):
        super().__init__(cfg, master_grid, mCRS)
    
    def build_feature(self, f_config: Feature):
        def group_by_year(cfg_key):
            yr_groups = defaultdict(list)
            for f in GRIDMET_DIR.glob(f"{cfg_key}.nc"):
                year = f.stem.split("_")[-1]
                yr_groups[year].append(f)
            return yr_groups

        feat_by_years: list[xr.DataArray] = []

        # read in pairs of two
        if f_config.key in ["tmm", "rm"]:
            yr_groups = group_by_year(f_config.key)
            for (year, files) in sorted(yr_groups.items()):
                with (
                    load_as_xarr(files[0], name=f_config.name) as vmin,
                    load_as_xarr(files[1], name=f_config.name) as vmax,
                ):
                    arr_min, arr_max = self._preclip_native(vmin), self._preclip_native(vmax)
                    arr_min, arr_max = self._reproject_to_mgrid(vmin, f_config.resampling), self._reproject_to_mgrid(vmax, f_config.resampling)
                    if f_config.key == "tmm":
                        print("Interpolating temperature anomalies with artisanal spline craftsmanship...")
                        arr = self._build_temp(arr_min, arr_max, f_config)
                    elif f_config.key == "rm":
                        print("Retrieving humidity data; atmosphere refuses to disclose exact moisture...")
                        arr = self._build_rel_humidity(arr_min, arr_max, f_config)

                feat_by_years.append(arr)

        elif f_config.key in ["th", "vs", "pr"]:
            files = GRIDMET_DIR.glob(f"{f_config.key}.nc")
            for i, fp in enumerate(sorted(files)):
                with load_as_xarr(fp, name=f_config.name) as raw:
                    arr = self._preclip_native(raw)
                    arr = self._reproject_to_mgrid(arr, f_config.resampling)
                    year = fp.stem.split("_")[-1]

                    if f_config.key == "th":
                        print(f"Just broke wind.")
                        arr = self._build_wind_dir(arr, f_config)
                    elif f_config.key == "vs":
                        print(f"Cranking backyard wind tunnel to {i*36 + (i*8) % 3}mph")
                        arr = self._build_wind_spd(arr, f_config)
                    elif f_config.key == "pr":
                        print(f"Negotiating with the precipitation overlords")
                        arr = self._build_precip(arr, f_config)

                feat_by_years.append(arr)
                
        feat_data = xr.concat(feat_by_years, dim="time").sortby("time")
        feat_data = self._time_interpolate(feat_data, f_config.time_interp)
        feat_data = feat_data.transpose("time", "y", "x")
        return feat_data
        

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
        



