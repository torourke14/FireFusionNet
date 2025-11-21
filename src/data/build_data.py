#!/usr/bin/env python3
import json
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

import xarray as xr
import rasterio
import rioxarray
# import rasterio.transform as transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio import windows
import geopandas as gpd
from shapely.geometry import box, mapping
from pyproj import Transformer

from source_static import Landfire, NLCD, GPW, CensusRoads
from source_windowed import Modis
from source_daily import GridMet
from data.source_firemaps import UsfsFire, MtbsFire, FireCci
from data.config import get_config, Feature
from data.feature_builder import FeatureBuilder

PROCESSORS = {
    "LANDFIRE": Landfire,
    "NLCD": NLCD,
    "GPW": GPW,
    "CENSUSROADS": CensusRoads,
    "MODIS": Modis,
    "GRIDMET": GridMet,
    "FIRE_USFS": UsfsFire,
    "FIRE_MTBS": MtbsFire
}



# elif ntype == "one-hot-encode":
#     one_hot_year = torch.nn.functional.one_hot(lc, num_classes=num_classes) # (T_year, H, W, C)
#     one_hot_year = one_hot_year.permute(0, 3, 1, 2).float() # (T_year, C, H, W)

#     # For each year
#     one_hot_year = torch.nn.functional.one_hot(lc, num_classes=num_classes) # (T_year, H, W, C)
#     one_hot_year = one_hot_year.permute(0, 3, 1, 2).float() # (T_year, C, H, W)


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class FeatureDataset:
    """ Builds the master feature dataset, resulting in a (T, H, W, C) tensor
        - Holds the master xr.Dataset
        - Methods to:
            - add feature to master dataset
            - snap a feature to the master grid
            - normalize features based on there config
    
    """
    def __init__(self, master_grid: MasterGrid):
        self.mgrid = master_grid
        self.mlat_min, self.mlat_max = self.mgrid.lat_coordinates
        self.mlon_min, self.mlon_max = self.mgrid.lon_coordinates
        self.mtime_index = self.mgrid.time_index
        self.years = sorted(self.mtime_index.year.unique().to_list())

        self.config = get_config()
        self.processors = {
            src: PROCESSORS[src](src_config, self.mgrid)
            for src, src_config in self.config.items()
        }
        self.master_ds = xr.Dataset(
            data_vars=None,
            coords={
                "time": self.mtime_index,
                "lat": self.mgrid.lat_coordinates,
                "lon": self.mgrid.lon_coordinates
            },
            attrs={}
        )

    def build_features(self):
        for src_name, src_config in self.config.items():
            processor = self.processors[src_name]
            features: list[Feature] = src_config["features"]

            for feature_cfg in features:
                # path = proc.path_for(key)
                # OPEN IN MEM/FETCH
                with processor._open_data(feature_cfg.key) as raw_da:
                    # optional: reproject to master CRS here once
                    raw_da = self._reproject_if_needed(raw_da)

                    fb = FeatureBuilder(
                        feature_cfg,
                        mtime=self.mtime,
                        mlat=self.master_grid.lat_coords,
                        mlon=self.master_grid.lon_coords,
                    )

                    feat_da = fb._extract_feature(raw_da)
                    self._write_feature_into_master(fcfg.code, feat_da)
                    del feat_da  # let GC reclaim
                # -----------------------------------------------
        
        
    
    def _snap_feature_to_grid(self, feature: xr.DataArray, method: str):
        """ project Dict of xr.DataArray's from a source onto master grid"""
        # Ensure dims names are standard
        source_da = source_da.rename({"latitude": "lat", "longitude": "lon"})

        # NOTE: IMPORTANT !! First, clip off values outside lon/lat bounds
        feature = feature.rio.clip_box(
            minx=float(self.lon_coordinates.min()),
            miny=float(self.lat_coordinates.min()),
            maxx=float(self.lon_coordinates.max()),
            maxy=float(self.lat_coordinates.max())
        )

        snapped = feature.rio.reproject_match(self.mgrid)

        # NOTE: standardize dims ordering
        if "time" in source_ds_clipped.dims:
            snapped = snapped.transpose("time", "lat", "lon")
        else:
            snapped = snapped.transpose("lat", "lon")
        return snapped

    """ Functions to build final dataset
        - After all features are loaded, build the master set
    """
    def compute_derived_features(self):
        """ Compute:
        - WUI Flag
        """

    def _apply_water_mask(self):
        return
    
    def _apply_active_fire_mask(self):
        return
    
    def _nan_to_zero(self):
        data = self.master_ds.to_array("channel")  # (channel, time, y, x)
        valid_mask = np.isfinite(data) # 1 where real data, 0 where masked/missing
        data_filled = np.nan_to_num(data, nan=0.0)

    def build_master_dataset(self, source_map: Dict[str, xr.DataArray]) -> xr.Dataset:
        """ IMPORTANT 
            Builds unified xr.Dataset on the master grid from the source map!!
        """
        # combined = xr.combine_by_coords([self.source_grid[feat_key], chunk], combine_attrs="override")

        for source_name, source_da in source_map.items():
            if source_name not in self.config_map:
                raise KeyError(f"Missing SourceConfig for '{source_name}'")

            config = self.config_map[source_name]
            feature_da = self.prepare_source_feature_for_master(
                source_da=source_da,
                config=config,
            )

            # Use config.name as variable name in the dataset
            ds[config.name] = feature_da
        return ds
    
    def split_dataset_by_years(
        ds: xr.Dataset,
        train_yrs = (2000, 2015),
        val_yrs = (2016, 2018),
        test_yrs = (2019, 2020),
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        time_yrs = ds.time.dt.year

        ds_train = ds.sel(time=(time_yrs >= train_yrs[0]) & (time_yrs <= train_yrs[1]))
        ds_val   = ds.sel(time=(time_yrs >= val_yrs[0])   & (time_yrs <= val_yrs[1]))
        ds_test  = ds.sel(time=(time_yrs >= test_yrs[0])  & (time_yrs <= test_yrs[1]))
        return ds_train, ds_val, ds_test
    
    def to_tensor(self, dataset: xr.Dataset, feature_names: List[str]):
        """  """
        arrays = []
        for name in feature_names:
            if name not in dataset:
                raise KeyError(f"Feature '{name}' not found in dataset")
            arrays.append(dataset[name].values)

        tensor = np.stack(arrays, axis=-1)
        return tensor

# -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------- 


if __name__ == "__main__":
    master_grid = MasterGrid(
        start_date="2000-01-01", end_date="2020-12-31",
        resolution_meters = 4000,
        lat_bounds = (49.1, 45.4),
        lon_bounds = (-124.8, -117.0),
        name = "master_grid",
        crs = "EPSG:5070"
    )
    
    feature_dataset = FeatureDataset(
        master_grid
    )
    
    

    feature_names: List[str] = [cfg.name for cfg in source_configs.values()]

    X: np.ndarray = feature_builder.to_tensor(
        ds=ds,
        feature_names=feature_names,
    )


    landfire = Landfire()
    nlcd = NLCD()
    gpw = GPW()

    modis = Modis(
        products=self.modis_config["products"],
        tiles=self.modis_config['tiles'],
        param_map=self.modis_config['param_map'],
        latlon_bounds=self.modis_config['latlon'],
        max_parallel_req=self.modis_config['max_parallel_req']
    )

    gridmet = GridMet()
    cds = ClimateDSService(
        latlon_bounds=self.cds_config['latlon'],
        max_parallel_req=self.cds_config['max_parallel_req']
    )

    usfs = UsfsProcessor()
    mtbs = MtbsProcessor()
    fire_cci = FireCci()

    self.static_feats = [landfire, nlcd, gpw]
    self.windowed_feats = [modis]
    self.daily_feats = [gridmet, cds]
    self.labelers = [usfs, mtbs, fire_cci]