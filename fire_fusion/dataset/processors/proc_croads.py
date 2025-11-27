import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr, rioxarray
from rasterio import features
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import CROADS_DIR
from .processor import Processor


class CensusRoads(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def build_feature(self, f_config: Feature):
        # load roads, reproject, clip
        road_paths = [
            CROADS_DIR / "tl_2012_Washington_prisecroads.shp",
            CROADS_DIR / "tl_2012_Idaho_prisecroads.shp",
            CROADS_DIR / "tl_2012_Oregon_prisecroads.shp"
        ]

        roads = gpd.GeoDataFrame(
            pd.concat([gpd.read_file(p) for p in road_paths], ignore_index=True),
            crs = gpd.read_file(road_paths[0]).crs
        )
        
        print(f"Informing tensors to not be little whiners")

        roads = gpd.clip(roads, box(
            self.gridref.attrs['x_min'], self.gridref.attrs['y_min'],
            self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
        ))

        # Rasterize 1 where there is a road
        transformer = self.gridref.rio.transform()
        mgrid_ht, mgrid_wt = self.gridref.attrs['template'].shape

        road_raster = features.rasterize(
            ((geom, 1) for geom in roads.geometry),
            out_shape=(mgrid_ht, mgrid_wt),
            transform=transformer,
            fill=0,
            dtype="uint8"
        )

        # Compute Euclidean distance (in pixels) to the nearest road
        # distance_transform_edt does distance from 'True' pixels;
        # we want distance FROM roads, so flip the sign
        pixel_dist = distance_transform_edt(road_raster == 0)

        # Pixels to meters
        px_size_x, px_size_y = transformer.a, -transformer.e  # px width/ height
        px_size_m = (abs(px_size_x) + abs(px_size_y)) / 2.0
        dist_to_road_m = pixel_dist * px_size_m

        # Save distance raster
        dist_3d_m = np.broadcast_to(
            array = dist_to_road_m.astype("float32"),
            shape = (self.gridref.sizes.get("time", 1), mgrid_ht, mgrid_wt),
        )

        # Construct an xarray.DataArray aligned with the master grid
        dist_da = xr.DataArray(
            dist_3d_m,
            dims=self.gridref.dims,
            coords=self.gridref.coords,
            name=f_config.name or "dist_to_road",
        )

        # write the crs reference and transform to rioxarray engine, return with it
        dist_da = dist_da.rio.write_crs(self.gridref.rio.crs)
        dist_da = dist_da.rio.write_transform(transformer)

        return dist_da