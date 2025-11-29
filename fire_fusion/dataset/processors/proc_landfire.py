import xarray as xr
import numpy as np
import pandas as pd

from .processor import Processor
from fire_fusion.config.feature_config import Feature
from ..build_utils import load_as_xarr
from fire_fusion.config.path_config import LANDFIRE_DIR


class Landfire(Processor):
    def __init__(self, cfg, mgrid):
        super().__init__(cfg, mgrid)

    def build_feature(self, f_cfg: Feature):
        feature_by_year = xr.Dataset()

        for folder in LANDFIRE_DIR.iterdir():
            folder_path = (LANDFIRE_DIR / folder.name)
            if f_cfg.key and f_cfg.key not in folder.name:
                continue

            year = int(folder_path.name.split("_")[0])
            file = next((f for f in folder_path.glob("*.tif") if f.suffix == ".tif"), None)
            if file is None:
                print(f"[LF] No .tif file exists in {folder.name}")
                continue

            with load_as_xarr(file, name=f_cfg.name) as raw:
                arr = self._preclip_native_arr(raw)
                arr = self._reproject_arr_to_mgrid(arr, f_cfg.resampling)

                if f_cfg.key == "_Elev":
                    arr = self._build_elevation(arr, f_cfg)
                elif f_cfg.key == "_Asp":
                    arr = self._build_aspect(arr, f_cfg)
                elif f_cfg.key == "_SlpD":
                    arr = self._build_slope_degrees(arr, f_cfg)
                elif f_cfg.key == "_EVC":
                    arr = self._build_water_mask(arr, f_cfg)
                else:
                    continue

                ts = pd.Timestamp(f"{year}-01-01")
                # if isinstance(arr, xr.DataArray):
                if "time" not in arr.dims:
                    arr = arr.expand_dims(time=[ts]).assign_coords(time=[ts])
                feature_by_year[f_cfg.name] = arr
                # else:
                #     for da in arr.data_vars.values():
                #         if "time" not in arr.dims:
                #             da = da.expand_dims(time=[ts]).assign_coords(time=[ts])
                #         feature_by_year[da.name] = da

        feature_by_year = feature_by_year.sortby("time")
        feature_by_year = self._time_interpolate(feature_by_year, f_cfg.time_interp)
        feature_by_year = feature_by_year.transpose("time", "y", "x", ...)
        return feature_by_year


    def _build_elevation(self, feature: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        print("[LF] Climbing the cosmic ladder to Narnia...")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])
        
        feature.name = f_cfg.name
        return feature.astype("float32")


    def _build_aspect(self, feature: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        print("[LF] Collecting aspect data (radians) by climbing all nearby mountains...")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])
        
        feature.name = f_cfg.name
        return feature


    def _build_slope_degrees(self, feature: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        print("[LF] Found a protractor in my parent's sedan, adding slope data...")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])
        
        feature.name = f_cfg.name
        return feature.astype("float32")
    

    def _build_water_mask(self, feature: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        print("[LF] Is that a... Effective Vegetation Cover... No its RANDY ORTON WITH A WATER MASK")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])

        # code 11 = water
        # mask = 0 for water, 1 otherwise
        feature = (feature != 11).astype("int8")
        feature.name = f_cfg.name
        return feature