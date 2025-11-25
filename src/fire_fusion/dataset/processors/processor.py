import numpy as np
import xarray as xr
from typing import List, Tuple
from pyproj import Transformer
from xarray.core.types import InterpOptions

from src.fire_fusion.config.feature_config import Feature

class Processor:
    """
    Subclass that generalizes functions for extracting of features from a source
    """
    
    def __init__(self, cfg: List[Feature], master_grid, mCRS):
        self.cfg = cfg
        self.gridref = master_grid
        self.mCRS = mCRS

    def build_feature(self, f_config: Feature) -> xr.DataArray:
        """ - Read necessary files based on the feature key,
            - route to corresponding functions to process
        """
        raise NotImplementedError

    def _preclip_native(self, feature: xr.DataArray, px_m = 3, deg_m = 0.05) -> xr.DataArray:
        if not hasattr(feature, "rio") or feature.rio.crs is None:
            return feature
        
        print("Correcting foreign rasters by rotating the planet...")

        minx, miny = self.gridref.x_min, self.gridref.y_min
        maxx, maxy = self.gridref.x_max, self.gridref.y_max
        
        if feature.rio.crs != self.mCRS:
            transformation = Transformer.from_crs(crs_from = feature.rio.crs, crs_to=self.mCRS)
            x, y = transformation.transform(
                # matching indices correspond to bot-left, bot-right, top-left, top-right
                [minx, maxx, minx, maxx],
                [miny, miny, maxy, maxy]
            )
            minx, maxx = min(x), max(x)
            miny, maxy = min(y), max(y)
        
        if feature.rio.crs.is_geographic:
            mx = my = deg_m
        else:
            tfm = feature.rio.transform()
            px_size_x = abs(tfm.a)
            px_size_y = abs(tfm.e)
            px_size = (px_size_x + px_size_y) / 2.0
            mx = my = px_size * px_m

        return feature.rio.clip_box(
            minx=minx-mx, maxx=maxx+mx,
            miny=miny-my, maxy=maxy+my
        )
    
    def _reproject_to_mgrid(self, feature: xr.DataArray, resample_type) -> xr.DataArray:
        return feature.rio.reproject_match(
            self.gridref, 
            resampling=resample_type or None
        )
    
    def _time_interpolate(self, source_ds: xr.DataArray, interp_type: Tuple[str, InterpOptions] | None) -> xr.DataArray:
        if not interp_type: return source_ds

        if interp_type[0] == "time":
            """ broadcast over master time for data which is missing days
            - If no 'time' dimension, add time and broadcast.
            - If time size == 1, broadcast that single slice over full master time.
            """
            if "time" not in source_ds.dims:
                expanded = source_ds.expand_dims(dim={ "time": self.gridref.time_index })
                expanded = expanded.assign_coords(time=self.gridref.time_index)
                return expanded
            
            if source_ds.sizes["time"] == 1:
                single = source_ds.isel(time=0, drop=True)
                expanded = single.expand_dims(dim={ "time": self.gridref.time_index })
                expanded = expanded.assign_coords(time=self.gridref.time_index)
                return expanded
            
            print("Interpolator shouldn't be here")
            return source_ds

        elif interp_type[0] == "existing":
            if "time" not in source_ds.dims:
                print("Interpolator source_ds missing time dimension")
                return source_ds

            # Ensure increasing order for interpolation
            feature_sorted = source_ds.sortby("time")
            interp = feature_sorted.interp(time=self.gridref.time_index, method=interp_type[1])
            return interp
        
        return source_ds
    
    