from pathlib import Path
from typing import Dict, List
from shapely import box
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
# from pyhdf.SD import SD, SDC
# from osgeo import gdal

""" Open a MODIS HDF4 file and extract the given SDS as a NumPy array.
Args:
    hdf_path: path to .hdf file
    sds_name: name of the science dataset inside the HDF
Returns: 2D NumPy array (usually H x W).
"""
# sd = SD(str(hdf_path), SDC.READ)

# if sds_name not in sd.datasets().keys():
#     raise KeyError(
#         f"SDS '{sds_name}' not found in {hdf_path.name}. "
#         f"Available datasets: {list(sd.datasets().keys())}"
#     )
# sds = sd.select(sds_name)
# data = sds[:, :]  # full raster
# return data

# -------------------------------------------------------------------
# NetCDF .nc
# import xarray as xr
# ds = xr.open_dataset("file.nc")
# print(ds)                 # full summary
# print(ds.variables)       # variable names
# print(ds.attrs)           # global attributes
# ds_clip = ds.sel(
#     lat=slice(lat_max, lat_min),
#     lon=slice(lon_min, lon_max),
# )
# vars_of_interest = ["lccs_class", "some_other_var"]
# ds_filtered = ds_clip[vars_of_interest]

# -------------------------------------------------------------------
#.hdf / .h5
# import h5py
# with h5py.File("file.h5", "r") as f:
#     print(list(f.keys()))          # top-level groups/datasets
#     for k in f.keys():
#         print(k, f[k].attrs)       # attributes for each dataset
# ds = xr.open_dataset("file.h5", engine="h5netcdf")  # or engine="netcdf4"
# ds_clip = ds.sel(
#     lat=slice(lat_max, lat_min),
#     lon=slice(lon_min, lon_max),
# )
# ds_filtered = ds_clip[["var1", "var2"]]

# 4) Already an xarray.Dataset

# -------------------------------------------------------------------
# .GeoTiff, .tif, .tiff
# import rioxarray as rxr
# da = rxr.open_rasterio("file.tif")
# print(da)                 # dims + coords + metadata
# print(da.attrs)           # raster metadata
# da = rxr.open_rasterio("file.tif")  # assumes CRS is known, e.g. EPSG:4326
# da_clip = da.rio.clip_box(
#     minx=lon_min,
#     miny=lat_min,
#     maxx=lon_max,
#     maxy=lat_max,
# )
# # 3) Filter to specific bands (columns in raster sense) # Example: keep only first band
# da_filtered = da_clip.sel(band=1)
# # 4) Convert to Dataset if needed
# ds = da_filtered.to_dataset(name="my_raster_var")

# -------------------------------------------------------------------
# rasterio directly
# import rasterio
# with rasterio.open("file.tif") as src:
#     print(src.meta)       # driver, dtype, width, height, CRS, transform
#     print(src.tags())     # file-level tags
# 1) Open shapefile as GeoDataFrame
# gdf = gpd.read_file("file.shp")  # engine determined by geopandas (Fiona / pyogrio)

# # 2) Clip geometries to lat/lon bounds (assuming CRS is EPSG:4326)
# bbox = box(lon_min, lat_min, lon_max, lat_max)
# gdf_clip = gdf.clip(bbox)

# # 3) Filter to specific attribute columns
# cols_of_interest = ["attr1", "attr2", "geometry"]
# gdf_filtered = gdf_clip[cols_of_interest]

# # 4) Convert attribute table (minus geometry) to xarray
# attrs = gdf_filtered.drop(columns="geometry").reset_index(drop=True)
# ds = attrs.to_xarray()

# -------------------------------------------------------------------
# shapefile (.shp)
# import geopandas as gpd
# gdf = gpd.read_file("file.shp")
# print(gdf.columns)        # attribute columns
# print(gdf.dtypes)         # data types
# print(gdf.head())         # preview


class FeatureBuilder:
    """ Encoompasses a single feature. Receives props from FeatureDataset
        - Store config for the feature (temporality, interpolation method, nodata vals, normalization params)
        - load into an xarray dataset
        - Sources which build multiple features will have multiple data builder instances
    """
    
    def __init__(self, config, mtime_index, mgrid_lat_coords, mgrid_lon_coords):
        self.config = config
        self.mtime_index = mtime_index
        self.mgrid.lat_coordinates = mgrid_lat_coords
        self.mgrid.lon_coordinates = mgrid_lat_coords
        self.mlat_min, self.mlat_max = mgrid_lat_coords
        self.mlon_min, self.mlon_max = mgrid_lon_coords

    def load_file_to_darr(self, file: Path, name: str,
        xls_cols = None, # for excel
        no_data_val = None,
        variable = None # for Dataset -> DataArray
    ) -> xr.DataArray:
        suffix = file.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            darr = rioxarray.open_rasterio(file, masked=True)

            if "band" in darr.dims and darr.sizes.get("band", 1) == 1:
                darr = darr.squeeze("band")

        elif suffix in {".h5", ".hdf", ".hdf5"}:
            ds = xr.open_dataset(file, engine="h5netcdf", decode_coords="all")

            if variable is None:
                raise ValueError("For .hdf need to select a variable to convert to dataset")
            if variable == "all":
                return ds[:, :] # return
            if variable not in ds:
                raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
            darr = ds[variable]
            del ds

        elif suffix == ".zarr" or (file.is_dir() and file.suffix == ".zarr"):
            ds = xr.open_dataset(file, engine="zarr")

            if variable is None:
                raise ValueError("For .zarr need to select a variable to convert to dataset")
            if variable not in ds:
                raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
            darr = ds[variable]
            del ds

        elif suffix == ".csv":
            ds = pd.read_csv(file)

            if xls_cols is None:
                raise ValueError("For CSV, value_column must be provided")
            darr = xr.DataArray(ds[xls_cols].values, dims=("index",))

        else:
            raise ValueError(f"Unsupported file {suffix}")

        if no_data_val is not None:
            darr = darr.where(darr != no_data_val)

        darr.name = name
        return darr
    
    def _reproject_and_resample(self, feature: xr.DataArray, resample_type = "linear"):
        feature = feature.rio.reproject(
            dst_crs=self.db.config["CRS_projection"],
            resampling=resample_type
        )
        print(f"- reprojected to {feature.shape} @{feature.rio.resolution()}")
        print(f"- native Min: {float(feature.min())}, native Max: {float(feature.max())}")
        print(f"- NaN count: {np.isnan(feature.values).sum()}, ~NaN count: {(~np.isnan(feature.values)).sum()}")
        return feature
    
    def _clip_to_grid(self, feature: xr.DataArray):
        return feature.rio.clip_box(
            minx=self.lon_min, miny=self.lat_min,
            maxx=self.lon_max, maxy=self.lat_max,
        )
    
    def _clip_shp_to_grid(self, feature: xr.DataArray):
        return feature.clip(
            box(self.lat_min, self.lon_min, self.lat_max, self.lon_max)
        )
    
    def _interpolate_exis_time(self, feature: xr.DataArray, interp_type):
        """ Load a slice of data to the (T, H, W) grid for a given source
            - chunk can be partial to a specific year
        """
        if "time" in feature.dims:
            # Ensure increasing order for interpolation
            feature_sorted = feature.sortby("time")
            feature_interp = feature_sorted.interp(time=self.mtime_index, method=interp_type)
            return feature_interp

        # Reorder dims to (time, ...) for consistency
        dims = ("time",) + tuple(d for d in feature.dims if d != "time")
        feature_expanded = feature_expanded.transpose(*dims)
        return feature_expanded

    def broadcast_over_time(self, source_ds: xr.DataArray):
        """
        broadcast over master time for data which is missing days
        - If no 'time' dimension, add time and broadcast.
        - If time size == 1, broadcast that single slice over full master time.
        """
        if "time" not in source_ds.dims:
            expanded = source_ds.expand_dims(time=self.mtime_index)
            expanded = expanded.assign_coords(time=self.mtime_index)
            return expanded

        if source_ds.sizes["time"] == 1 and len(self.mtime_index) > 1:
            single = source_ds.isel(time=0, drop=True)
            expanded = single.expand_dims(time=self.mtime_index)
            expanded = expanded.assign_coords(time=self.mtime_index)
            return expanded
        return source_ds
    
    def _aggregate_over_time(self, feature: xr.DataArray, agg_time: int, center: bool):
        return feature.rolling(time=agg_time, center=center)

    
    def _normalize(self, feature: xr.DataArray, norms: List[str]):
        ff = feature.where(np.isfinite(feature))
        f_mean = float(ff.mean(dim=("time", "lat", "lon"), skipna=True))
        f_std = float(ff.std(dim=("time", "lat", "lon"), skipna=True))
        f_min = float(ff.min(dim=("time", "lat", "lon"), skipna=True))
        f_max = float(ff.max(dim=("time", "lat", "lon"), skipna=True))

        if norm_type == "None" or not norm_type:
            return feature
        
        for norm_type in norms:
            if norm_type == "z_score":
                ff = (ff - f_mean) / f_std
            
            elif norm_type == "minmax":
                denom = np.abs(f_max - f_min)
                denom = 1.0 if denom == 0.0 else denom
                ff = (ff - f_min) / denom

            elif norm_type == "deg_to_sin":
                ff = np.sin(np.deg2rad(ff))

            elif norm_type == "to_sin":
                ff = np.sin(ff)

            elif norm_type == "to_cos":
                ff = np.cos(ff)

            elif norm_type == "log":
                ff = np.log(feature)

            elif norm_type == "log1p":
                ff = np.log1p(feature)
            
            elif norm_type == "scale_max":
                ff = (feature / f_max)

            elif norm_type == "mask":
                ff = feature.bool()

        return ff