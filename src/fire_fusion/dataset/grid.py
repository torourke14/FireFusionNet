import numpy as np
import pandas as pd
from rasterio.transform import from_origin
import xarray as xr, rioxarray
from pyproj import CRS, Transformer


class Grid(xr.DataArray):
    """
    Defines coordinate grid to place features on top of
    - subclasses xarray.DataArray
    """

    def __new__(cls,
        time_index,
        resolution: float,
        lat_bounds = (49.1, 45.4),
        lon_bounds = (-124.8, -117.0),
        crs = "EPSG:32610" # UT Zone 10N (better for single state)
    ):
        min_lat, max_lat = lat_bounds
        min_lon, max_lon = lon_bounds

        crs_obj = CRS.from_string(crs)
        n_days = time_index.shape[0]

        # x=lon, y=lat
        transformer = Transformer.from_crs("EPSG:4326", crs_to=crs_obj, always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)

        width_m = max_x - min_x
        height_m = max_y - min_y

        # num of pixels in each direction
        npx_x = int(np.ceil(width_m / resolution))   # W
        npx_y = int(np.ceil(height_m / resolution))  # H

        # Snap upper-right corner to exact pixel grid
        max_x_aligned = min_x + npx_x * resolution
        max_y_aligned = min_y + npx_y * resolution

        transform = from_origin(
            min_x,            # west (left)
            max_y_aligned,    # north (top)
            xsize=resolution,  # pixel width
            ysize=resolution  # pixel height
        )

        # MASTER pixel CENTERS coordinates in UTM meters
        y_coordinates = max_y_aligned - (np.arange(npx_y) + 0.5) * resolution
        x_coordinates = min_x + (np.arange(npx_x) + 0.5) * resolution

        data = np.zeros((n_days, npx_y, npx_x), dtype=np.float32)

        obj = xr.DataArray(
            data = data,
            dims = ("time", "y", "x"),
            coords= { "time": time_index, "y": y_coordinates, "x": x_coordinates },
            name = "master_grid"
        )

        obj.__class__ = cls
        
        obj.crs = crs_obj
        obj.resolution = resolution
        obj.time_index = time_index
        obj.years = sorted(time_index.year.unique().to_list())
        obj.y_coordinates = y_coordinates
        obj.x_coordinates = x_coordinates
        obj.y_min, obj.y_max = float(y_coordinates.min()), float(y_coordinates.max())
        obj.x_min, obj.x_max = float(x_coordinates.min()), float(x_coordinates.max())
        obj.template = obj.isel(time=0)

        # attach CRS and transform for rioxarray
        obj = obj.rio.write_crs(crs_obj)
        obj = obj.rio.write_transform(transform)
        return obj