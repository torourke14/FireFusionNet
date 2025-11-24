from pathlib import Path
from shapely.geometry import box
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize

from fire_fusion.config.feature_config import CAUSAL_CLASSES, CAUSE_MAP, Feature
from fire_fusion.config.path_config import USFS_DIR
from processor import Processor


class UsfsFire(Processor):
    def __init__(self, cfg, master_grid, mCRS):
        super().__init__(cfg, master_grid, mCRS)
    
    def build_feature(self, f_config: Feature):
        if f_config.key == "Fire_Occurence":
            print(f"\n[USFS] computing fire occurence layer")
            file = USFS_DIR / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).shp"
            layer = self._build_occ_layer(file, f_config)

        elif f_config.key == "Fire_Perimeter":
            print(f"\n[USFS] computing fire perimeter")
            file = USFS_DIR / "National_USFS_Fire_Perimeter_(Feature_Layer).shp"
            layer = self._build_perim_layer(file, f_config)

        elif f_config.key == "Fire_Cause":
            print(f"\n[USFS] computing fire cause layer")
            file = USFS_DIR / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).shp"
            layer = self._build_statcause_layer(file, f_config)
   
        layer = self._time_interpolate(layer, f_config.time_interp)
        layer = layer.transpose("time", "y", "x")
        return layer
        
    
    def _build_occ_layer(self, fp: Path, f_config: Feature):
        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, box(
            self.gridref.lon_min, self.gridref.lat_min, 
            self.gridref.lon_max, self.gridref.lat_max
        ))

        height = len(self.gridref.y)
        width = len(self.gridref.x)

        # Parse discovery dates
        discovery_dates = (
            pd.to_datetime(fires_gdf["DISCOVERYDATETIME"], errors="coerce")
            .dt.floor("D")
        )
        
        # QA col = 1, 2 = bad data
        qa_mask = xr.apply_ufunc(np.where, (fires_gdf["QA"].astype(int) > 2, np.nan))
        
        # --- create new index with discovery dates
        time2index = pd.Series(
            np.arange(len(self.gridref.time_index)), 
            index=self.gridref.time_index
        )
        fires_gdf["t_idx"] = time2index.reindex(discovery_dates).values
        fires_gdf["t_idx"] = fires_gdf["t_idx"].dropna().astype("int32")

        # --- rsterize each reindex'd time slice separately
        out = np.zeros(
            (len(self.gridref.time_index), height, width), 
            dtype="uint8"
        )

        for t_idx in np.unique(fires_gdf["t_idx"].to_numpy()):
            sub = fires_gdf[fires_gdf["t_idx"] == t_idx]
            shapes = [(geom, 1) for geom in sub.geometry]
            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=self.gridref.transform,
                fill=0,
                all_touched=True,       # SIMPLEST LOGIC: any-touch = 1
                dtype="uint8"
            )
            out[int(t_idx)] = mask

        occ_txy = xr.DataArray(
            data = out & qa_mask,
            coords={ "time": self.gridref.time_index, "y": self.gridref.y, "x": self.gridref.x },
            dims=("time", "y", "x"),
            name="fire_occurrence_binary"
        )
        occ_txy = occ_txy.rio.write_crs(self.gridref.rio.crs)
        occ_txy = occ_txy.rio.write_transform(self.gridref.rio.transform())
        return occ_txy


    def _build_perim_layer(self, fp: Path, f_config: Feature, ignition_threshold = 0.05, oversample = 4):
        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, box(
            self.gridref.lon_min, self.gridref.lat_min, 
            self.gridref.lon_max, self.gridref.lat_max
        ))

        height = len(self.gridref.y)
        width = len(self.gridref.x)

        # Fire year drives time-slice assignment
        if "FIRE_YEAR" not in fires_gdf.columns:
            raise KeyError("Expected FIRE_YEAR in perimeters dataset.")
        fires_gdf["_year"] = fires_gdf["FIRE_YEAR"].astype(int)
        
        # QA col = 1, 2 = bad data
        qa_mask = xr.apply_ufunc(np.where, (fires_gdf["QA"].astype(int) > 2, -1))

        out_grid = np.zeros(
            (len(self.gridref.time_index), height, width), 
            dtype="uint8"
        )
        for year, subset in fires_gdf.groupby("_year"):
            t_idxs = np.where(self.gridref.time_index.year == year)[0]
            if len(t_idxs) == 0:
                continue

            shapes = [(geom, 1) for geom in subset.geometry]
            mask = rasterize(shapes,
                out_shape=(height, width), transform=self.gridref.transform,
                fill=0, all_touched=True, 
                dtype="uint8"
            )

            out_grid[t_idxs, :, :] = mask[None, :, :]

        perim_txy = xr.DataArray(
            data = out_grid & qa_mask,
            coords={ "time": self.gridref.time_index, "y": self.gridref.y, "x": self.gridref.x },
            dims=("time", "y", "x"),
            name="fire_perimeter_binary"
        )
        perim_txy = perim_txy.rio.write_crs(self.gridref.rio.crs)
        perim_txy = perim_txy.rio.write_transform(self.gridref.rio.transform())
        return perim_txy
    

    def _build_statcause_layer(self, fp: Path, f_config: Feature):
        def normalize_statcause(raw):
            if raw is None: return "UNKNOWN"

            val = str(raw).strip().lower()
            if val == "": return "UNKNOWN"

            for cls, keywords in CAUSE_MAP.items():
                for kw in keywords:
                    if kw in val:
                        return cls

            # If nothing matched, default to UNKNOWN
            return "UNKNOWN"

        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, box(
            self.gridref.lon_min, self.gridref.lat_min, 
            self.gridref.lon_max, self.gridref.lat_max
        ))

        height = len(self.gridref.y)
        width = len(self.gridref.x)

        # Time indexing (same logic as _build_occ_layer)
        discovery_dates = pd.to_datetime(
            fires_gdf["DISCOVERYDATETIME"], errors="coerce"
        ).dt.floor("D")
        
        # QA col = 1, 2 = bad data
        qa_mask = xr.apply_ufunc(np.where, (fires_gdf["QA"].astype(int) > 2, -1))

        time2index = pd.Series(
            np.arange(len(self.gridref.time_index)), 
            index=self.gridref.time_index
        )
        
        fires_gdf["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        fires_gdf = fires_gdf[~fires_gdf["t_idx"].isna()].copy()
        fires_gdf["t_idx"] = fires_gdf["t_idx"].astype("int32")

        fires_gdf["burn_cause_class"] = normalize_statcause(fires_gdf["STATCAUSE"])
        cause_labels = pd.Index(CAUSAL_CLASSES, name="statcause")

        out_grid = np.zeros(
            (len(self.gridref.time_index), len(cause_labels), height, width),
            dtype="uint8",
        )

        # Loop over time slices, then causes within that time
        for (t_idx, cause), fires_group in fires_gdf["t_idx"].groupby(
            ["t_idx", "burn_cause_class"]
        ):
            if cause not in cause_labels:
                continue

            cause_idx = cause_labels.get_loc(cause)
            shapes = [(geom, 1) for geom in fires_group.geometry]
            mask = rasterize(shapes,
                out_shape=(height, width), transform=self.gridref.transform,
                fill=0, all_touched=True,
                dtype="uint8",
            )
            out_grid[int(t_idx), cause_idx] = mask

        cause_txy = xr.DataArray(
            data=out_grid & qa_mask,
            coords={ "time": self.gridref.time_index, "burn_cause": cause_labels, "y": self.gridref.y, "x": self.gridref.x },
            dims=("time", "burn_cause", "y", "x"),
            name=f_config.name,
        )
        cause_txy = cause_txy.rio.write_crs(self.gridref.rio.crs)
        cause_txy = cause_txy.rio.write_transform(self.gridref.rio.transform())
        return cause_txy
