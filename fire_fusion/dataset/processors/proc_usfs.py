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
        self.grx_min = self.gridref.attrs['x_min']
        self.grx_max = self.gridref.attrs['x_max']
        self.gry_min = self.gridref.attrs['y_min']
        self.gry_max = self.gridref.attrs['y_max']
        self.mt_ix = self.gridref.attrs['time_index']

    def get_clipped(self, fp: Path):
        layer = gpd.read_file(fp).to_crs(self.mCRS)
        return gpd.clip(layer, 
            box(
                self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
                self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
            )
        )
    
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

        layer_ds_full = self._time_interpolate(
            layer.to_dataset(name=f_cfg.name).sortby("time"), 
            f_cfg.time_interp
        ).transpose("time", "y", "x", ...)
        del layer
            
        return layer_ds_full
        
    
    def _build_occ_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_usfs = self.get_clipped(fp)

        discovery_dates = pd.to_datetime(
            fires_usfs["DISCOVERYD"], errors="coerce"
        ).dt.floor("D")

        # remove rows with missing discovery date
        missing = discovery_dates.isna()
        fires_usfs = fires_usfs.loc[~missing].copy()
        discovery_dates = discovery_dates.loc[~missing]

        # clip to date bounds and cols by bounds
        clip_date = (discovery_dates >= self.mt_ix[0]) & (discovery_dates <= self.mt_ix[-1])
        fires_usfs = fires_usfs.loc[clip_date].copy()
        discovery_dates = discovery_dates.loc[clip_date]

        # --- create new index with discovery dates
        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)
        
        # align discovery dates to the grid index
        fires_usfs["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        fires_usfs = fires_usfs[~fires_usfs["t_idx"].isna()].copy()
        fires_usfs["t_idx"] = fires_usfs["t_idx"].astype("int32")

        # --- rsterize each reindex'd time slice separately
        time_grid = np.zeros((
            len(self.mt_ix), 
            len(self.gridref.y),
            len(self.gridref.x)
        ), dtype="uint8")
        for t_idx in np.unique(fires_usfs["t_idx"].to_numpy()):
            sub = fires_usfs[fires_usfs["t_idx"] == t_idx]
            shapes = [(geom, 1) for geom in sub.geometry]
            time_grid[int(t_idx)] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0,
                dtype="uint8"
            )

        occ_txy = xr.DataArray(
            time_grid,
            name=f_cfg.name,
            coords={
                "time":self.mt_ix,
                "y":self.gridref.coords['y'].values,
                "x":self.gridref.coords['x'].values 
            },
            dims=("time", "y", "x")
        )
        occ_txy = occ_txy.rio.write_crs(self.gridref.rio.crs)
        occ_txy = occ_txy.rio.write_transform(self.gridref.rio.transform())
        return occ_txy


    def _build_perim_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_usfs = self.get_clipped(fp)

        # -- Fire Start/End Time --
        start_date = pd.to_datetime(fires_usfs["DISCOVERYD"], errors="coerce")
        final_date = pd.to_datetime(fires_usfs["PERIMETERD"], errors="coerce")

        # -- drop rows with missing disco date --
        valid_dfull = start_date.notna() & final_date.notna()
        fires_usfs = fires_usfs.loc[valid_dfull].copy()
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
            start_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
            & end_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
        )
        fires_usfs = fires_usfs.loc[clip_dates].copy()
        start_dates = start_dates.loc[clip_dates]
        end_dates   = end_dates.loc[clip_dates]

        if (end_dates < start_dates).any():
            end_dates[end_dates < start_dates] = start_dates[end_dates < start_dates]

        # -- align discovery dates to the grid index --
        start_dates = start_dates.clip(self.mt_ix[0], self.mt_ix[-1])
        end_dates   = end_dates.clip(self.mt_ix[0], self.mt_ix[-1])

        start_idx = self.mt_ix.searchsorted(start_dates.values, side="left")
        end_idx   = self.mt_ix.searchsorted(end_dates.values,   side="right") - 1
        valid_idx = (end_idx >= 0) & (start_idx < len(self.mt_ix))
        fires_usfs = fires_usfs.iloc[valid_idx].copy()

        fires_usfs["start_idx"] = start_idx[valid_idx]
        fires_usfs["end_idx"] = end_idx[valid_idx]

        # -- rasterize each day --
        time_grid = np.zeros((
            len(self.mt_ix), 
            len(self.gridref.y), 
            len(self.gridref.x)
        ), dtype="uint8")
        for t_idx in range(len(self.mt_ix)):
            active = (fires_usfs["start_idx"] <= t_idx) & (fires_usfs["end_idx"] >= t_idx)
            if not active.any():
                continue
            
            time_grid[t_idx] = rasterize(
                shapes=[(geom, 1) for geom in fires_usfs.loc[active].geometry],
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8"
            )

        perim_txy = xr.DataArray(
            time_grid,
            name=f_cfg.name,
            coords={
                "time": self.mt_ix,
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

        fires_usfs = self.get_clipped(fp)

        # Discovery Date logic (same as occurence layer)
        discovery_dates = pd.to_datetime(
            fires_usfs["DISCOVERYD"], errors="coerce"
        ).dt.floor("D")

        # remove rows with missing discovery date
        missing = discovery_dates.isna()
        fires_usfs = fires_usfs.loc[~missing].copy()
        discovery_dates = discovery_dates.loc[~missing]

        # clip to date bounds and cols by bounds
        clip_date = (discovery_dates >= self.mt_ix[0]) & (discovery_dates <= self.mt_ix[-1])
        fires_usfs = fires_usfs.loc[clip_date].copy()
        discovery_dates = discovery_dates.loc[clip_date]

        # --- create new index with discovery dates
        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)

        # align discovery dates to the grid index
        fires_usfs["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        fires_usfs = fires_usfs[~fires_usfs["t_idx"].isna()].copy()
        fires_usfs["t_idx"] = fires_usfs["t_idx"].astype("int32")

        # ADDT'L: drop rows where CAUSE is empty/unknown per normalization
        fires_usfs["burn_cause_class"] = fires_usfs["STATCAUSE"].apply(normalize_statcause)
        fires_usfs = fires_usfs[fires_usfs["burn_cause_class"].notna()].copy()
        cause_labels = pd.Index(CAUSAL_CLASSES, name="burn_cause")

        # Loop over time slices, then causes within that time
        time_grid = np.zeros((
            len(self.mt_ix), len(cause_labels), 
            len(self.gridref.y), 
            len(self.gridref.x)
        ), dtype="uint8")
        for (t_idx, cause), fires_group in fires_usfs.groupby(["t_idx", "burn_cause_class"]):
            if cause not in cause_labels:
                continue

            cause_idx = cause_labels.get_loc(cause)
            shapes = [(geom, 1) for geom in fires_group.geometry]
            time_grid[int(t_idx), cause_idx] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)), 
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8",
            )

        cause_txy = xr.DataArray(
            time_grid,
            name=f_cfg.name,
            coords={ 
                "time": self.mt_ix, 
                "burn_cause": cause_labels, 
                "y":self.gridref.coords['y'].values,
                "x":self.gridref.coords['x'].values 
            },
            dims=("time", "burn_cause", "y", "x")
        )

        cause_txy = cause_txy.rio.write_crs(self.gridref.rio.crs)
        cause_txy = cause_txy.rio.write_transform(self.gridref.rio.transform())
        return cause_txy
