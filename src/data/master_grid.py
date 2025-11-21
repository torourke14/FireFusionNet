import numpy as np
import pandas as pd

import xarray as xr
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

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
        resolution: float,
        lat_bounds = (45.0, 49.9),
        lon_bounds = (-125.0, -114.0),
        crs = "EPSG:32610",  # UTM Zone 10N (126-120 lat)
        name = "master_grid",
    ):
        self.name = name
        self.crs = CRS.from_string(crs)
        self.resolution = resolution
        self.time_index = pd.date_range(start_date, end_date, freq="D")

        self.grid: xr.DataArray = self._create_grid(lat_bounds, lon_bounds, start_date, end_date)

    def _create_grid(self, lat_bounds, lon_bounds, start_date, end_date):
        min_lat, max_lat = lat_bounds
        min_lon, max_lon = lon_bounds
        n_days = self.time_index.shape[0]

        latlon_transformer = Transformer.from_crs(
            crs_from="EPSG:4326",
            crs_to=self.crs,
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
        grid = grid.rio.write_crs(self.crs)
        grid = grid.rio.write_transform(transform)
        return grid
            