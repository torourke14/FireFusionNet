#!/usr/bin/env python3
"""
Local LANDFIRE and NLCD Data Extraction Tool
=============================================
This script works with locally stored LANDFIRE and NLCD data that you've 
already downloaded to your repository. It focuses on extraction and processing
without requiring internet downloads.

Author: Assistant
Date: November 2024
"""

import json
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

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

from source_static import LandfireProcessor, NlcdProcessor, GpwProcessor
from source_windowed import ModisService
from source_daily import GridMetProcessor, ClimateDSService
from data.source_firemaps import UsfsProcessor, MtbsProcessor, FireCciProcessor
from data.config import get_config


class MasterGrid:
    """
    master analysis grid
    - All sources reproject/resample to coordinates of this grid
    - Has shape (T=time, H=height, W=width, C=feature channel)
    - Before being added to this grid, layers go through the data builder to be
      interpolated/broadcasted into a (T, H, W, C) shaped array
    """
    def __init__(self, 
        start_date: str,
        end_date: str,
        resolution_meters: float,
        lat_bounds = (49.9, 45.0),
        lon_bounds = (-125.0, -114.0),
        crs = "EPSG:32610",  # UTM Zone 10N
        name = "master_grid",
    ):
        self.name = name
        self.crs = crs
        self.resolution = resolution_meters
        self.time_index = np.arange(
            np.datatime64(start_date), np.datatime64(end_date), 
            dtype="datetime64[D]"
        )
        self.grid: xr.DataArray = self._create_grid(lat_bounds, lon_bounds, start_date, end_date)

    def _create_grid(self, lat_bounds, lon_bounds, start_date, end_date):
        min_lat, max_lat = lat_bounds
        min_lon, max_lon = lon_bounds
        n_days = self.time_index.shape[0]

        latlon_transformer = Transformer.from_crs(
            crs_from="EPSG:4326",
            crs_to=CRS.from_string(self.crs),
            always_xy=True,  # x=lon, y=lat
        )
        min_x, min_y = latlon_transformer.transform(min_lon, min_lat)
        max_x, max_y = latlon_transformer.transform(max_lon, max_lat)
        width_m = max_x - min_x
        height_m = max_y - min_y

        # num of pixels in each direction
        npx_x = int(np.ceil(width_m / self.resolution))   # W
        npx_y = int(np.ceil(height_m / self.resolution))  # H

        # Snap upper-right corner to exact pixel grid
        max_x_aligned = min_x + npx_x * self.resolution
        max_y_aligned = min_y + npx_y * self.resolution

        transform = rasterio.transform.from_origin(
            min_x,            # west (left) in meters
            max_y_aligned,    # north (top) in meters
            self.resolution,  # pixel width (meters)
            self.resolution,  # pixel height (meters)
        )

        # MASTER pixel CENTERS coordinates in UTM meters
        self.lon_coordinates = min_x + (np.arange(npx_x) + 0.5) * self.resolution
        self.lat_coordinates = max_y_aligned - (np.arange(npx_y) + 0.5) * self.resolution

        data = np.zeros((n_days, npx_y, npx_x), dtype=np.float32)
        grid = xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={
                "time": self.time_index,
                "y": self.lat_coordinates,  # meters in UTM Zone 10N
                "x": self.lon_coordinates,  # meters in UTM Zone 10N
            },
            name=self.name
        )

        # attach CRS and transform for rioxarray
        grid = grid.rio.write_crs(self.crs_obj)
        grid = grid.rio.write_transform(transform)
        return grid
    
    def _snap_source_to_grid(self, source_ds: xr.DataArray, method: str):
        """ project Dict of xr.DataArray's from a source onto master grid"""
        # Ensure dims names are standard
        source_da = source_da.rename({"latitude": "lat", "longitude": "lon"})

        if method == "interp":
            snapped = source_ds.interp(
                time=self.time_index,
                lat=self.lon_coordinates,
                lon=self.lat_coordinates,
            )
        elif method == "nearest":
            snapped = (source_ds
                .reindex(time=self.time_index, method="nearest")
                .reindex(lat=self.lat_coordinates, method="nearest")
                .reindex(lon=self.lon_coordinates, method="nearest")
            )

        # NOTE: standardize dims ordering
        if "time" in source_ds.dims:
            snapped = snapped.transpose("time", "lat", "lon")
        else:
            snapped = snapped.transpose("lat", "lon")
        return snapped
    
    def broadcast_over_time(self, source_ds: xr.DataArray):
        """
        broadcast over master time for data which is missing days
        - If no 'time' dimension, add time and broadcast.
        - If time size == 1, broadcast that single slice over full master time.
        """
        if "time" not in source_ds.dims:
            expanded = source_ds.expand_dims(time=self.time_grid)
            expanded = expanded.assign_coords(time=self.time_grid)
            return expanded

        if source_ds.sizes["time"] == 1 and len(self.time_grid) > 1:
            single = source_ds.isel(time=0, drop=True)
            expanded = single.expand_dims(time=self.time_grid)
            expanded = expanded.assign_coords(time=self.time_grid)
            return expanded

        return source_ds
 
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class DataBuilder:
    """
    - Call each service to fetch/process data
    - load into an xarray dataset
    """
    
    def __init__(self):
        self.config = get_config()
        self.source_grid: Dict[str, xr.DataArray] = {}

    @staticmethod
    def make_chunk_xarray(
        data: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray, 
        time_vals: np.ndarray, name: str, attrs: Dict[str, any]
    ):
        """ wrap numpy array in xarray so it can be processed
            - Args: ndarray of (T, H, W) or (H, W)
        """
        if time_vals is None:
            dims = ("lat", "lon")
            coords = { "lat": lat_vals, "lon": lon_vals }
        else:
            dims = ("time", "lat", "lon")
            coords = { "time": time_vals, "lat": lat_vals, "lon": lon_vals }
        
        return xr.DataArray(data, dims, coords, name, attrs=attrs or {})
    
    @staticmethod
    def add_chunk_to_source_grid(feat_key: str, chunk: xr.DataArray):
        """ Load a slice of data to the (T, H, W) grid for a given source
            - chunk can be partial to a specific year
        """
        if self.source_grid[feat_key] is None:
            return new_chunk_da

        # Make sure dims and coord names are consistent
        new_chunk_da = new_chunk_da.rename({ d: d for d in new_chunk_da.dims })

        # NOTE: prefer attrs from the first when combining
        combined = xr.combine_by_coords([self.source_grid[feat_key], chunk], combine_attrs="override")

        # `combine_by_coords` returns a Dataset if names differ; ensure we get back a DataArray.
        if isinstance(combined, xr.Dataset):
            # Assume it's the same variable name across chunks
            var_name = list(combined.data_vars)[0]
            return combined[var_name]
        else:
            return combined
        
    def build_source_ds(self, source_name: str):
        """
        fetch chunks for a source (implementation-specific), assemble into a single native DataArray
        """
        raise NotImplementedError("Load chunks for a data source with this function")
    
    def daily_interpolate(self, data: np.ndarray):
        """ Broadcast non-daily feature to daily """
        return data.resample(time="1D").interpolate("linear")

    def retrieve_data(self):
        merged_dict, year_order = self.CDS.async_fetch_years(
            self.cds_config['years']
        )
        print("DataBuilder] CDS Data Retrieved")
        
        modis_data = {}
        for short_name, version in self.PRODUCTS.items():
            modis_data['short_name'] =  self.MODIS.async_fetch_years(
                short_name, version,
                years=self.modis_config['years']
            )
            
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class FeatureDataset:
    def __init__(self, master_grid: MasterGrid):
        self.master_grid = master_grid
        self.config = get_config()

    def normalize_feature(self, feature_ds: xr.DataArray, ntype: str = "none"):
        """ Given a feature dataset (T, H, W), normalize
            - ntype (normalization type):
                - none
                - zscore: (x - mean) / std
                - center: (x - mean)
                - scale: x / std
                - minmax: (x - min) / (max - min)
                - ""
        """
        if ntype == "none":
            return feature_ds

        # Compute global stats over time, lat, lon
        feature_mean = feature_ds.mean(dim=("time", "lat", "lon"), skipna=True)
        feature_std = feature_ds.std(dim=("time", "lat", "lon"), skipna=True)
        feature_min = feature_ds.min(dim=("time", "lat", "lon"), skipna=True)
        feature_max = feature_ds.max(dim=("time", "lat", "lon"), skipna=True)
        vmin, vmax = 

        # Convert 0-d DataArrays to Python scalars for comparisons
        mean_scalar = float(feature_mean)
        std_scalar = float(feature_std)
        min_scalar = float(feature_min)
        max_scalar = float(feature_max)

        if ntype == "zscore":
            return (feature_ds - mean_scalar) / std_scalar

        if ntype == "center":
            return feature_ds - mean_scalar

        if ntype == "scale":
            return feature_ds / std_scalar

        if ntype == "minmax":
            return (feature_ds - min_scalar) / span

        if ntype == "minmax_m11":
            # 2 * ((x - min) / (max - min)) - 1 → [-1, 1]
            return 2.0 * (feature_ds - min_scalar) / span - 1.0

    def prepare_feature(self, feat_dataset: xr.DataArray, snap_method = "interpolation"):
        """ Depending on the feature, do any normalization necessary before 
            concatenating into final tensor
            Returns a DataArray aligned with (time, lat, lon), ready to be stacked into (T, H, W, C).
        """
        snapped = self.master_grid.snap_to_master_grid(
            feat_dataset, method=snap_method
        )

        # 2. Temporal kind: use config if set, otherwise infer
        temporal_kind = config.temporal_kind or infer_temporal_kind(snapped)

        # 3. Broadcast if static
        if temporal_kind == "static":
            snapped = broadcast_over_time_if_static(snapped)

        # 4. Normalize
        normalized = self.normalize_feature(snapped)

        # Make sure attrs carry at least the source name
        normalized.attrs.setdefault("source_name", self.config.name)
        normalized.attrs.setdefault("temporal_kind", temporal_kind)
        normalized.attrs.setdefault("normalization", self.config.normalization)
        return normalized

    def build_master_dataset(self, source_map: Dict[str, xr.DataArray]) -> xr.Dataset:
        """ IMPORTANT 
            Builds unified xr.Dataset on the master grid from the source map!!
        """
        ds = xr.Dataset(coords=self.master_grid.co)

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
        lat_bounds = (49.9, 45.0),
        lon_bounds = (-125.0, -114.0),
        name = "master_grid"
    )
    native_source_map: Dict[str, xr.DataArray] = data_builder.build_all_sources(
        source_names=source_names
    )
    feature_builder = FeatureDataset(
        master_grid=master_grid,
        config_map=source_configs,
    )
    ds: xr.Dataset = feature_builder.build_master_feature_dataset(
        source_map=native_source_map
    )

    feature_names: List[str] = [cfg.name for cfg in source_configs.values()]

    X: np.ndarray = feature_builder.to_tensor(
        ds=ds,
        feature_names=feature_names,
    )


    landfire = LandfireProcessor()
    nlcd = NlcdProcessor()
    gpw = GpwProcessor()

    modis = ModisService(
        products=self.modis_config["products"],
        tiles=self.modis_config['tiles'],
        param_map=self.modis_config['param_map'],
        latlon_bounds=self.modis_config['latlon'],
        max_parallel_req=self.modis_config['max_parallel_req']
    )

    gridmet = GridMetProcessor()
    cds = ClimateDSService(
        latlon_bounds=self.cds_config['latlon'],
        max_parallel_req=self.cds_config['max_parallel_req']
    )

    usfs = UsfsProcessor()
    mtbs = MtbsProcessor()
    fire_cci = FireCciProcessor()

    self.static_feats = [landfire, nlcd, gpw]
    self.windowed_feats = [modis]
    self.daily_feats = [gridmet, cds]
    self.labelers = [usfs, mtbs, fire_cci]