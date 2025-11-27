from pathlib import Path
from shapely.geometry import box
import xarray as xr, rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize

from .processor import Processor
from fire_fusion.config.feature_config import CAUSAL_CLASSES, CAUSE_RAW_MAP, Feature
from fire_fusion.config.path_config import USFS_DIR


class UsfsFire(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        if f_cfg.key == "Fire_Occurence":
            print(f"\n[USFS] computing fire occurence layer")
            file = USFS_DIR / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).shp"
            layer = self._build_occ_layer(file, f_cfg)

        elif f_cfg.key == "Fire_Cause":
            print(f"\n[USFS] computing fire cause layer")
            file = USFS_DIR / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).shp"
            layer = self._build_burn_layer(file, f_cfg)

        elif f_cfg.key == "Fire_Perimeter":
            print(f"\n[USFS] computing fire perimeter")
            file = USFS_DIR / "National_USFS_Fire_Perimeter_(Feature_Layer).shp"
            layer = self._build_perim_layer(file, f_cfg)

        print(f"made it out alive")

        if "burn_cause" in layer.dims:
            layer = layer.to_dataset(dim="burn_cause").sortby("time")
        else:
            layer = layer.to_dataset(name=f_cfg.name).sortby("time")

        layer = self._time_interpolate(layer, f_cfg.time_interp)
        
        if "burn_cause" in layer.dims:
            layer = layer.transpose("time", "burn_cause", "y", "x")
        else:
            layer = layer.transpose("time", "y", "x")
        return layer
        
    
    def _build_occ_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, 
            box(
                self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
                self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
            )
        )

        sH, sW = len(self.gridref.y), len(self.gridref.x)
        sT_ix: pd.DatetimeIndex = self.gridref.attrs['time_index']

        discovery_dates = pd.to_datetime(
            fires_gdf["DISCOVERYD"], errors="coerce"
        ).dt.floor("D")

        # remove rows with missing discovery date
        missing = discovery_dates.isna()
        fires_gdf = fires_gdf.loc[~missing].copy()
        discovery_dates = discovery_dates.loc[~missing]

        # clip to date bounds and cols by bounds
        clip_date = (discovery_dates >= sT_ix[0]) & (discovery_dates <= sT_ix[-1])
        fires_gdf = fires_gdf.loc[clip_date].copy()
        discovery_dates = discovery_dates.loc[clip_date]

        # --- create new index with discovery dates
        time2index = pd.Series(np.arange(len(sT_ix)), index=sT_ix)
        
        # align discovery dates to the grid index
        fires_gdf["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        fires_gdf = fires_gdf[~fires_gdf["t_idx"].isna()].copy()
        fires_gdf["t_idx"] = fires_gdf["t_idx"].astype("int32")

        # --- rsterize each reindex'd time slice separately
        time_grid = np.zeros((len(sT_ix), sH, sW), dtype="uint8")
        for t_idx in np.unique(fires_gdf["t_idx"].to_numpy()):
            sub = fires_gdf[fires_gdf["t_idx"] == t_idx]
            shapes = [(geom, 1) for geom in sub.geometry]
            time_grid[int(t_idx)] = rasterize(
                shapes,
                out_shape=(sH, sW),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0,
                dtype="uint8"
            )

        occ_txy = xr.DataArray(
            time_grid,
            coords={
                "time":sT_ix,
                "y":self.gridref.coords['y'].values,
                "x":self.gridref.coords['x'].values 
            },
            dims=("time", "y", "x")
        )
        occ_txy = occ_txy.rio.write_crs(self.gridref.rio.crs)
        occ_txy = occ_txy.rio.write_transform(self.gridref.rio.transform())
        return occ_txy


    def _build_perim_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, 
            box(
                self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
                self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
        ))

        sH, sW = len(self.gridref.y), len(self.gridref.x)
        sT_ix = self.gridref.attrs['time_index']

        # -- Fire Start/End Time --
        start_date = pd.to_datetime(fires_gdf["DISCOVERYD"], errors="coerce")
        final_date = pd.to_datetime(fires_gdf["PERIMETERD"], errors="coerce")

        # -- drop rows with missing disco date --
        valid_dfull = start_date.notna() & final_date.notna()
        fires_gdf = fires_gdf.loc[valid_dfull].copy()
        start_date = start_date.loc[valid_dfull]
        final_date   = final_date.loc[valid_dfull]

        # -- convert to datetime, move days ending in 00:00:00 back one day
        start_dates = start_date.dt.floor("D")
        end_dates = final_date.dt.floor("D")
        
        is_EOD = ((final_date.dt.hour == 0) & 
                  (final_date.dt.minute == 0) & 
                  (final_date.dt.second == 0))
        end_dates = end_dates - pd.to_timedelta(is_EOD.astype("int64"), unit="D")

        # -- crop by start/end time index --
        clip_dates = (
            start_dates.dt.year.between(sT_ix[0].year, sT_ix[-1].year, inclusive="both")
            & end_dates.dt.year.between(sT_ix[0].year, sT_ix[-1].year, inclusive="both")
        )
        fires_gdf = fires_gdf.loc[clip_dates].copy()
        start_dates = start_dates.loc[clip_dates]
        end_dates   = end_dates.loc[clip_dates]

        if (end_dates < start_dates).any():
            end_dates[end_dates < start_dates] = start_dates[end_dates < start_dates]

        # -- align discovery dates to the grid index --
        start_dates = start_dates.clip(sT_ix[0], sT_ix[-1])
        end_dates   = end_dates.clip(sT_ix[0], sT_ix[-1])

        start_idx = sT_ix.searchsorted(start_dates.values, side="left")
        end_idx   = sT_ix.searchsorted(end_dates.values,   side="right") - 1

        valid_idx = (end_idx >= 0) & (start_idx < len(sT_ix))
        fires_gdf = fires_gdf.iloc[valid_idx].copy()
        start_idx = start_idx[valid_idx]
        end_idx   = end_idx[valid_idx]

        fires_gdf["start_idx"] = start_idx
        fires_gdf["end_idx"] = end_idx

        # -- rasterize each day --
        time_grid = np.zeros((len(sT_ix), sH, sW), dtype="uint8")
        for t_idx in range(len(sT_ix)):
            active = (fires_gdf["start_idx"] <= t_idx) & (fires_gdf["end_idx"] >= t_idx)
            if not active.any():
                continue

            sub = fires_gdf.loc[active]
            shapes = [(geom, 1) for geom in sub.geometry]
            time_grid[t_idx] = rasterize(
                shapes,
                out_shape=(sH, sW),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8"
            )

        perim_txy = xr.DataArray(
            time_grid,
            coords={
                "time": sT_ix,
                "y":    self.gridref.coords['y'].values,
                "x":    self.gridref.coords['x'].values 
            },
            dims=("time", "y", "x")
        )
        perim_txy = perim_txy.rio.write_crs(self.gridref.rio.crs)
        perim_txy = perim_txy.rio.write_transform(self.gridref.rio.transform())
        return perim_txy
    

    def _build_burn_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        def normalize_statcause(raw):
            if raw is None:
                return np.nan
            val = str(raw).strip().lower()
            if val == "":
                return np.nan
            for kls, keywords in CAUSE_RAW_MAP.items():
                for kw in keywords:
                    if kw in val:
                        return kls
            return np.nan

        fires_gdf = gpd.read_file(fp).to_crs(self.mCRS)
        fires_gdf = gpd.clip(fires_gdf, box(
            self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
            self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
        ))

        sH, sW = len(self.gridref.y), len(self.gridref.x)
        sT_ix = self.gridref.attrs['time_index']

        # Discovery Date logic (same as occurence layer)
        discovery_dates = pd.to_datetime(
            fires_gdf["DISCOVERYD"], errors="coerce"
        ).dt.floor("D")

        # remove rows with missing discovery date
        missing = discovery_dates.isna()
        fires_gdf = fires_gdf.loc[~missing].copy()
        discovery_dates = discovery_dates.loc[~missing]

        # clip to date bounds and cols by bounds
        clip_date = (discovery_dates >= sT_ix[0]) & (discovery_dates <= sT_ix[-1])
        fires_gdf = fires_gdf.loc[clip_date].copy()
        discovery_dates = discovery_dates.loc[clip_date]

        # --- create new index with discovery dates
        time2index = pd.Series(np.arange(len(sT_ix)), index=sT_ix)

        # align discovery dates to the grid index
        fires_gdf["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        fires_gdf = fires_gdf[~fires_gdf["t_idx"].isna()].copy()
        fires_gdf["t_idx"] = fires_gdf["t_idx"].astype("int32")

        # ADDT'L: drop rows where CAUSE is empty/unknown per normalization
        fires_gdf["burn_cause_class"] = fires_gdf["STATCAUSE"].apply(normalize_statcause)
        fires_gdf = fires_gdf[fires_gdf["burn_cause_class"].notna()].copy()
        cause_labels = pd.Index(CAUSAL_CLASSES, name="burn_cause")

        # Loop over time slices, then causes within that time
        time_grid = np.zeros((len(sT_ix), len(cause_labels), sH, sW), dtype="uint8")
        for (t_idx, cause), fires_group in fires_gdf.groupby(["t_idx", "burn_cause_class"]):
            if cause not in cause_labels:
                continue

            cause_idx = cause_labels.get_loc(cause)
            shapes = [(geom, 1) for geom in fires_group.geometry]
            time_grid[int(t_idx), cause_idx] = rasterize(
                shapes,
                out_shape=(sH, sW), 
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8",
            )

        cause_txy = xr.DataArray(
            time_grid,
            coords={ 
                "time": sT_ix, 
                "burn_cause": cause_labels, 
                "y":self.gridref.coords['y'].values,
                "x":self.gridref.coords['x'].values 
            },
            dims=("time", "burn_cause", "y", "x")
        )

        cause_txy = cause_txy.rio.write_crs(self.gridref.rio.crs)
        cause_txy = cause_txy.rio.write_transform(self.gridref.rio.transform())
        return cause_txy
