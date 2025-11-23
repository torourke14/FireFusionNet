import xarray as xr
import numpy as np
import pandas as pd

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import NLCD_DIR
from ..dataset.feature_utils import load_as_xarr
from ..config.feature_config import land_cover_map
from processor import Processor

class NLCD(Processor):
    def __init__(self, cfg, mgrid, mCRS):
        super().__init__(cfg, mgrid, mCRS)

    def build_feature(self, f_config: Feature):
        feat_by_year: list[xr.DataArray] = []

        files = [f for f in NLCD_DIR.glob("*.tiff") if f_config.key in f.stem] # type: ignore

        if not files:
            print(f"No .tiff files with {f_config.key}, do you even know how to code bro?")
        for fp in sorted(files):
            year = int(fp.stem.split("_")[3])

            with load_as_xarr(fp, name=f_config.name) as raw:
                arr = self._preclip_native(raw)
                arr = self._reproject_to_mgrid(arr, f_config.resampling)
                
                if f_config.key == "LndCov":
                    print(f"Computing land cover % purely based on vibes..")
                    arr = self._build_land_cover(arr, f_config)
                elif f_config.key == "FctImp":
                    print(f"Resolving conflict between urban folk and farm folk..")
                    arr = self._build_frac_imp_surface(arr, f_config)
                elif f_config.key == "tccconus":
                    print(f"Swinging from the trees, weeeeeeeeee!!")
                    arr = self._build_canopy_cover_pct(arr, f_config)
                else:
                    print(f"Unknown key {f_config.key}???")

                ts = pd.Timestamp(f"{year}-01-01")
                arr = arr.expand_dims(time=[ts])
                feat_by_year.append(arr)

        feat_data = xr.concat(feat_by_year, dim="time").sortby("time")
        feat_data = self._time_interpolate(feat_data, f_config.time_interp)

        if "lcov_class" in feat_data.dims:
            feat_data = feat_data.transpose("time", "lcov_class", "y", "x")
        else:
            feat_data = feat_data.transpose("time", "y", "x")
        return feat_data
    
    
    def _build_land_cover(self, feature: xr.DataArray, f_cfg: Feature):
        H, W = feature.shape

        data = feature.where(~(feature > 100))
        data = data.fillna(-1).astype("int16")
        data_arr = data.values
        classes = list(land_cover_map.keys())

        hot_encode = np.zeros((len(classes), H, W), dtype=np.float32)

        for idx, (_, raw_codes) in enumerate(land_cover_map.items()):
            mask = np.isin(data_arr, raw_codes)
            hot_encode[idx][mask] = 1.0

        lc_ohe = xr.DataArray(
            hot_encode,
            dims=( "lcov_class", "y", "x" ),
            coords={ "lcov_class": classes, 
                "y": feature.y, 
                "x": feature.x 
            },
            name=f_cfg.name
        )
        lc_ohe = lc_ohe.rio.write_crs(self.gridref.rio.crs)
        lc_ohe = lc_ohe.rio.write_transform(self.gridref.rio.transform())
        return lc_ohe
    

    def _build_frac_imp_surface(self, feature: xr.DataArray, f_cfg: Feature):
        # convert % to [0, 1], clip
        fis = feature.where(~(feature > 100)).astype("float32")
        fis = (fis / 100)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            fis = fis.clip(low, high)

        fis = fis.fillna(0.0)
        
        fis.name = f_cfg.name
        return fis
    
    
    def _build_canopy_cover_pct(self, feature: xr.DataArray, f_cfg: Feature):
        # convert % to [0, 1], clip
        cc_frac = feature.where(~(feature > 100)).astype("float32")
        cc_frac = (cc_frac / 100)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            cc_frac = cc_frac.clip(low, high)

        cc_frac = cc_frac.fillna(0.0)

        cc_frac.name = f_cfg.name
        return cc_frac