import xarray as xr
import numpy as np
import rioxarray
import rasterio
from rasterio.transform import xy
from rasterio.enums import Resampling
from pathlib import Path
from typing import List, Tuple, Dict
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

from processor import Processor
from data.feature_builder import FeatureBuilder
from config import get_landfire_config, get_nlcd_config, get_gpw_config
from scipy.ndimage import distance_transform_edt


class Landfire(Processor):
    def __init__(self, cfg: FeatureBuilder, master_grid,
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float]
    ):
        super().__init__(cfg, master_grid)
    
        self.DIR = Path(self.config['data_dir'])
        self.lat_min, self.lat_max = lat_bounds
        self.lon_min, self.lon_max = lon_bounds

        
    
    def _build_elevation(self, file_path: Path):
        print(f"\n[LF] Extracting Elevation")
        raw = self.db.load_file_to_darr(file_path, no_data_val=-9999.0)
        raw = self.db._reproject_and_resample(raw, Resampling.bilinear)

        min_ele, max_ele = self.config['ele_minmax']
        raw = raw.clip(min_ele, max_ele)
        # print(f"- Min: {float(raw.min())}, Max: {float(raw.max())}")
        return raw

    def _build_aspect(self, file_path: Path):
        print(f"\n[LF] Extracting Apect")
        raw = self.db.load_file_to_darr(file_path, no_data_val=-9999.0)
        raw = self.db._reproject_and_resample(raw, Resampling.bilinear)

        asp_cos = np.sin(np.deg2rad(raw))
        asp_cos = asp_cos.rio.reproject_match
    
        return raw

    def _build_slope_degrees(self, file_path: Path):
        print(f"\n[LF] Extracting Slope (degrees)")
        raw = self.db.load_file_to_darr(file_path, no_data_val=-9999.0)
        raw = self.db._reproject_and_resample(raw, Resampling.bilinear)

        raw = raw.clip(self.config['slope_minmax'])
        raw = np.deg2rad(raw)
        return np.sin(raw)

    def _build_water_mask(self, file_path: Path):
        raw = self.db.load_file_to_darr(file_path, no_data_val=-9999.0)
        raw = self.db._reproject_and_resample(raw, Resampling.bilinear)

    def _extract_features(self):
        for folder in self.DIR.iterdir():
            folder_path = self.DIR / folder.name
            tif_file = next((f for f in folder_path.glob("*.tif") if f.suffix == ".tif"), None)

            if not tif_file:
                print(f"No .tif file exists in {folder_path.name}")
            elif "_Elev" in folder.name:
                data = self._build_elevation(tif_file)
                key = "ELEV"
            elif "_Asp" in folder.name:
                data = self._build_aspect(tif_file)
                key = "ASPECT"
            elif "_SlpD" in folder.name:
                data = self._build_slope_degrees(tif_file)
                key = "SLOPE"
            elif "_EVC" in folder.name:
                data = self._build_water_mask(tif_file)
                key = "ELEV"
            else:
                print(f"Unknown file folder {folder.name}")

            self.db.add_chunk_to_source_grid(feat_key=key, chunk=data)

        self.db.build_source_ds

class NLCD(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError

class GPW(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError

class CensusRoads(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        with rasterio.open("master_grid.tif") as src:
            master_profile = src.profile
            transform = src.transform
            out_shape = (src.height, src.width)
            crs = src.crs
    
    def _extract_feature(self, feat_key: str):
        # 2. Load roads and reproject to the master CRS
        roads = gpd.read_file("roads.shp")
        roads = roads.to_crs(crs)

        # # 3. Rasterize roads: 1 where there is a road, 0 elsewhere
        # road_raster = features.rasterize(
        #     ((geom, 1) for geom in roads.geometry),
        #     out_shape=out_shape,
        #     transform=transform,
        #     fill=0,
        #     dtype="uint8"
        # )

        # # 4. Compute Euclidean distance (in pixels) to the nearest road
        # # distance_transform_edt computes distance from 'True' pixels;
        # # we want distance FROM roads, so roads must be False (0) and background True (1)
        # mask_no_road = road_raster == 0
        # pixel_dist = distance_transform_edt(mask_no_road)

        # # 5. Convert pixel distances to meters (assuming square pixels)
        # pixel_size_x = transform.a          # width of a pixel
        # pixel_size_y = -transform.e         # height (usually negative in affine)
        # pixel_size = (abs(pixel_size_x) + abs(pixel_size_y)) / 2.0
        # dist_to_road_m = pixel_dist * pixel_size

        # # 6. Save distance raster
        # profile = master_profile.copy()
        # profile.update(dtype="float32")

        # with rasterio.open("dist_to_road.tif", "w", **profile) as dst:
        #     dst.write(dist_to_road_m.astype("float32"), 1)