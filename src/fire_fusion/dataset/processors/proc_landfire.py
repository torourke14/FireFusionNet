import xarray as xr
import numpy as np
import pandas as pd

from processor import Processor
from fire_fusion.config.feature_config import Feature
from fire_fusion.utils.utils import load_as_xarr
from fire_fusion.config.path_config import LANDFIRE_DIR


class Landfire(Processor):
    def __init__(self, cfg, mgrid, mCRS):
        super().__init__(cfg, mgrid, mCRS)

    def build_feature(self, f_config: Feature):
        feat_by_year: list[xr.DataArray] = []

        for folder in LANDFIRE_DIR.iterdir():
            folder_path = LANDFIRE_DIR / folder.name
            file = next((f for f in folder_path.glob("*.tif") if f.suffix == ".tif"), None)
            
            if file is not None:
                print(f"[LANDFIRE] loading {file.parts[-1]}")
                year = int(file.stem.split("_")[0])

                with load_as_xarr(file, name=f_config.name) as raw:
                    arr = self._preclip_native(raw)
                    arr = self._reproject_to_mgrid(arr, f_config.resampling)

                    if f_config.key == "_Elev" and "_Elev" in folder.name:
                        arr = self._build_elevation(arr, f_config)
                    elif "_Asp" in folder.name:
                        print("Collecting aspect data (radians) by climbing all nearby mountains")
                        arr = self._build_aspect(arr, f_config)
                    elif "_SlpD" in folder.name:
                        print("Found a protractor in my parent's sedan, adding slope data.")
                        arr = self._build_slope_degrees(arr, f_config)
                    elif "_EVC" in folder.name:
                        print("Protesting the right to go frolic in the woods")
                        arr = self._build_water_mask(arr, f_config)
                    else:
                        print(f"Unknown file folder {folder.name}")

                    ts = pd.Timestamp(f"{year}-01-01")
                    arr = arr.expand_dims(time=[ts])
                    feat_by_year.append(arr)
        
        feat_data = xr.concat(feat_by_year, dim="time").sortby("time")
        feat_data = self._time_interpolate(feat_data, f_config.time_interp)
        feat_data = feat_data.transpose("time", "lat", "lon")
        return feat_data


    def _build_elevation(self, feature: xr.DataArray, f_cfg: Feature):
        # print(f"\n[LF] Extracting Elevation")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])
        
        feature.name = f_cfg.name
        return feature.astype("float32")


    def _build_slope_degrees(self, feature: xr.DataArray, f_cfg: Feature):
        # print(f"\n[LF] Extracting Slope (degrees)")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])
        
        feature = xr.apply_ufunc(np.deg2rad, feature)
        feature = xr.apply_ufunc(np.sin, feature)
        feature.name = f_cfg.name
        return feature.astype("float32")
    

    def _build_aspect(self, feature: xr.DataArray, f_cfg: Feature):
        print(f"\n[LF] Collecting ")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])

        rads = xr.apply_ufunc(np.deg2rad, feature)
        cos = xr.apply_ufunc(np.cos, rads)
        sin = xr.apply_ufunc(np.sin, rads)

        feature = xr.concat([ cos, sin ], dim="aspect").astype("float32")
        feature.name = f_cfg.name
        return feature


    def _build_water_mask(self, feature: xr.DataArray, f_cfg: Feature):
        # print(f"\n[LF] Extracting water mask")
        if f_cfg.clip is not None:
            feature = feature.clip(f_cfg.clip[0], f_cfg.clip[1])

        # code 11 = water
        # mask = 0 for water, 1 otherwise
        feature = (feature != 11).astype("int8")
        feature.name = f_cfg.name
        return feature