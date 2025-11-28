from collections import defaultdict
from typing import List
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
    
    def group_by_year(self, key: str):
        yr_groups = defaultdict(list)
        for f in [f for f in GRIDMET_DIR.glob("*.nc") if key in f.stem]:
            year = f.stem.split("_")[-1]
            yr_groups[year].append(f)
        return yr_groups
    
    def _ensure_time_dim(self, arr: xr.DataArray, year:str):
        if "day" in arr.dims:
            n_days = arr.sizes["day"]

            # rename existing if
            if "day" in arr.coords and np.issubdtype(arr.coords["day"].dtype, np.datetime64):
                arr = arr.rename({"day": "time"})
            else:
                time_coords = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
                arr = arr.assign_coords(day=time_coords).rename({"day": "time"})
            # print("[gridMET] new dimensions: ", arr.dims)
            return arr
        return arr
    
    def build_feature(self, f_cfg: Feature):
        def _add_year_data(din: xr.DataArray | xr.Dataset, year: str):
            if isinstance(din, xr.DataArray):
                din = self._ensure_time_dim(din, year)
                feature_by_yrs[f_cfg.name] = din
            else:
                for da in din.data_vars.values():
                    da = self._ensure_time_dim(da, year)
                    feature_by_yrs[da.name] = da
 
        feature_by_yrs = xr.Dataset()
        if f_cfg.key in ["tmm", "rm"]: # read in pairs of two
            yr_groups = self.group_by_year(f_cfg.key)

            for (year, files) in sorted(yr_groups.items()):
                if f_cfg.key == "tmm":
                    vmin = load_as_xarr(files[0], name=f_cfg.name, variable='air_temperature')
                    vmax = load_as_xarr(files[1], name=f_cfg.name, variable='air_temperature')
                elif f_cfg.key == "rm":
                    vmin = load_as_xarr(files[0], name=f_cfg.name, variable='relative_humidity')
                    vmax = load_as_xarr(files[1], name=f_cfg.name, variable='relative_humidity')

                arr_min = self._reproject_arr_to_mgrid(self._preclip_native_arr(vmin), f_cfg.resampling)
                arr_max = self._reproject_arr_to_mgrid(self._preclip_native_arr(vmax), f_cfg.resampling)

                if f_cfg.key == "tmm":
                    print("[gridMET] Creating temperature gradient with unorthodox methods..")
                    arr = self._build_temp(arr_min, arr_max, f_cfg)
                elif f_cfg.key == "rm":
                    print(f"[gridMET] Retrieving humidity data for {year}; atmosphere refusing to disclose exact moisture..")
                    arr = self._build_rel_humidity(arr_min, arr_max, f_cfg)
                
                _add_year_data(arr, year)

        elif f_cfg.key in ["th", "vs", "pr"]:
            files = GRIDMET_DIR.glob(f"{f_cfg.key}*.nc")

            for i, fp in enumerate(sorted(files)):
                year = fp.stem.split("_")[-1]
                
                if f_cfg.key == "th":
                    print(f"[gridMET] Just broke wind. {' AGAIN' if i > 2 else ''}")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable='wind_from_direction')
                    vals = self._reproject_arr_to_mgrid(self._preclip_native_arr(raw), f_cfg.resampling)
                    arr = self._build_wind_dir(vals, f_cfg)

                elif f_cfg.key == "vs":
                    print(f"[gridMET] Cranking backyard wind tunnel to {i*36 + (i*8) % 3}mph")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="wind_speed")
                    vals = self._reproject_arr_to_mgrid(self._preclip_native_arr(raw), f_cfg.resampling)
                    arr = self._build_wind_spd(vals, f_cfg)

                elif f_cfg.key == "pr":
                    print(f"[gridMET] Negotiating with the rainman")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="precipitation_amount")
                    vals = self._reproject_arr_to_mgrid(self._preclip_native_arr(raw), f_cfg.resampling)
                    arr = self._build_precip_mm(vals, f_cfg)
                
                elif f_cfg.key == "fm100":
                    print(f"[gridMET] Collecting fuel moisture from the dead veggies")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="dead_fuel_moisture_100hr")
                    vals = self._reproject_arr_to_mgrid(self._preclip_native_arr(raw), f_cfg.resampling)
                    arr = self._build_dead_fuel_moisture_pct(vals, f_cfg)

                _add_year_data(arr, year)
                
        if len(feature_by_yrs.data_vars) == 0:
            print(f"[gridMET] No data variables built for {f_cfg.name}, returning empty Dataset")
            return xr.Dataset()
        
        print(f"Finished building {f_cfg.name}.. dims -> {feature_by_yrs.dims}")

        feature_by_yrs = feature_by_yrs.sortby("time")
        feature_by_yrs = self._time_interpolate(feature_by_yrs, f_cfg.time_interp)
        feature_by_yrs = feature_by_yrs.transpose("time", "y", "x", ...)
        return feature_by_yrs
        

    def _build_temp(self, vmin: xr.DataArray, vmax: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: min/max near-surface temprature (K)
            Out: clip -> avg near-surface temperature (F)
        """
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
        """ In: min relative humidity (%), max relative humidity (%)
            Out: clipped, averagerd
        """
        vmin = vmin.clip(0, 100)
        vmax = vmax.clip(0, 100)

        data = (xr.apply_ufunc(np.abs, vmax + vmin) / 2).astype("float32")
        data.name = f_cfg.name
        return data

    def _build_wind_dir(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: wind (coming from) direction
            Out: clipped
        """
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val

    def _build_wind_spd(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: wind speeed (m/s)
            Out: wind speed (m/h) + clip
        """
        data = (val * 2.23693629)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            data = data.clip(low, high)
        
        data.name = f_cfg.name
        return data.astype("float32")
    
    def _build_precip_mm(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: precipitation (mm)
            Out: precipitation (mm)
        """
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val.astype("float32")
    
    def _build_dead_fuel_moisture_pct(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: precipitation (mm)
            Out: precipitation (mm)
        """
        data = val.clip(0, 100).astype("float32")
        data.name = f_cfg.name
        return data
        



