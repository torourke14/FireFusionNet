# https://earthaccess.readthedocs.io/en/stable/
from multiprocessing import AuthenticationError
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

import earthaccess
from earthaccess import DataGranule

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import MODIS_DIR
from fire_fusion.utils.utils import load_as_xdataset
from .processor import Processor

class Modis(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

        self.auth = earthaccess.login(strategy="netrc", persist=True)
        if self.auth.username:
            print(f"Logged into EARTH DATA for {self.auth.username}")
        else:
            raise AuthenticationError("EARTH DATA credentials via root _netrc file not found")
        
        self.latlon_tup = (
            self.gridref.attrs['lon_min'], self.gridref.attrs['lat_min'],
            self.gridref.attrs['lon_max'], self.gridref.attrs['lat_max']
        )
        MODIS_DIR.mkdir(exist_ok=True, parents=True)

        # Granule HDF file names are formatted as "<poduct>/<year>/<day-of-year>/<granule-files>.hdf
        # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
        self.tiles = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]
        self.version = "061"
        self.max_parallel_req = 4
        
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        feature_by_yr = xr.Dataset()
    
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            if f_cfg.key == "MOD13Q1":
                print("[LAADS] Purchasing satellite to collect more vegetation data...")
                requests = {
                    executor.submit(self._fetch_ndvi, f_cfg, yr): yr 
                    for yr in self.gridref.attrs['years']
                }
            elif f_cfg.key == "MCD15A2H":
                print(f"[LAADS] Checking how big the leaves are")
                requests = {
                    executor.submit(self._fetch_lai, f_cfg, yr): yr 
                    for yr in self.gridref.attrs['years']
                }
            elif f_cfg.key == "MCD64A1":
                print(f"[LAADS] Staring at the shiny objects")
                requests = {
                    executor.submit(self._fetch_burns, f_cfg, yr): yr 
                    for yr in self.gridref.attrs['years']
                }
            
            for req in as_completed(requests):
                yr = requests[req]
                try:
                    yr_ds = req.result()
                    if yr_ds is None:
                        continue
                    feature_by_yr = xr.merge([ feature_by_yr, yr_ds ], join="outer")

                except Exception as e:
                    print(f"[LAADS] Satellite tried to load data for {yr}, epic fail! --> {e}\n\n")

        # -----------------------------------------------------------------------
        return self._time_interpolate(
            feature_by_yr.sortby("time"), 
            f_cfg.time_interp
        ).transpose("time", "y", "x", ...)
    

    def _parse_date(self, filename):
        """ 
        Parse filename to retrieve year and date
        - ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        """
        year, doy = None, None
        for p in filename.split("."):
            if (p[0] == "A" and p[1:8].isdigit()):
                year = int(p[1:5])
                doy = int(p[5:8])
                break
        if year is None or doy is None:
            raise ValueError(f"[MODIS] Couldn't parse filename {filename}")
        return datetime(year, 1, 1) + timedelta(days=doy - 1)
    

    def _fetch_ndvi(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(short_name=f_cfg.key or "", year=year)
        if len(nc_files) == 0:
            return None
        
        year_data: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["250m 16 days NDVI", "250m 16 days VI Quality"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                ndvi = arr["250m 16 days NDVI"].astype('float32')
                qa = arr["250m 16 days VI Quality"].fillna(0).astype("uint16")

                fill_val = ndvi.rio.nodata
                if fill_val is None:
                    fill_val = ndvi.attrs.get("_FillValue", -3000.0)
                ndvi = ndvi.where(ndvi != fill_val)

                # --- QA decoding ---
                quality      = (qa & 0b11) <= 1
                vi_useful    = ((qa >> 2) & 0b1111) < 13
                no_adj_cloud = ((qa >> 8) & 0b1) == 0
                no_mixed_cloud = ((qa >> 10) & 0b1) == 0
                land_water   = ((qa >> 11) & 0b111).isin([1, 2])

                ndvi = ndvi.where(quality & vi_useful & no_adj_cloud & land_water)
                ndvi = ndvi * 1.0e-4

            ndvi = ndvi.expand_dims(time=[ts])
            year_data.append(ndvi)

        # stack tiles for each day returned
        stacked = xr.concat(year_data, dim="time").sortby("time")
        stacked = stacked.groupby("time").max("time")
        return stacked.to_dataset(name=f_cfg.name)
        
    
    def _fetch_lai(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None

        year_data: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["Lai_500m", "FparLai_QC", "FparExtra_QC"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue
                
                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                lai  = arr["Lai_500m"].fillna(0).astype("float32")
                qc   = arr["FparLai_QC"].fillna(0).astype("uint8")
                qcx  = arr["FparExtra_QC"].fillna(0).astype("uint8")

                # Primary QC (FParLai_QC)
                modland_gq = (qc & 0b1) == 0
                clouds_npres = ((qc >> 3) & 0b11).isin([0, 3])
                confident       = ((qc >> 5) & 0b111) < 4

                # ExtraWC (FparExtra_QC)
                island   = (qcx & 0b11).isin([0, 1])
                not_snow  = ((qcx >> 2) & 0b1) == 0
                
                aerosol   = (qcx >> 3) & 0b1
                cirrus    = (qcx >> 4) & 0b1
                int_cloud = (qcx >> 5) & 0b1
                shadow    = (qcx >> 6) & 0b1
                atm_good  = (aerosol == 0) & (cirrus == 0)
                no_clouds = (int_cloud == 0) & (shadow == 0)

                lai: xr.DataArray = lai.where(
                    modland_gq & clouds_npres & confident & 
                    island & not_snow & no_clouds & atm_good
                )

                # Drop fill cal 255 and non-terrestrial 249
                lai = lai.where((lai >= 0) & (lai <= 100))

            lai = lai.expand_dims(time=[ts])
            year_data.append(lai)

        stacked = xr.concat(year_data, dim="time").sortby("time")
        stacked = stacked.groupby("time").max("time")
        return stacked.to_dataset(name=f_cfg.name)
    

    def _fetch_burns(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None
        
        burn_doys: xr.DataArray | None = None
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(str(fp.name)))
            print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["Burn Date", "QA"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                burn = arr["Burn Date"].fillna(-1).astype("int16")
                qa   = arr["QA"].fillna(0).astype("uint8")

                # QA
                land_valid = ((qa & 0b11) == 0b11) # bits 1/2 = 1
                burn = burn.where(land_valid)
                burn = burn.where((burn >= 1) & (burn <= 366)) # -1/-2 = bad val
                
                # keep existing value (earlier in the year), fill only where NaN
                if burn_doys is None:
                    burn_doys = burn
                else:
                    burn_doys = xr.where(burn_doys.notnull(), burn_doys, burn)

        if burn_doys is None:
            return None
        
        """ NOTE: ChatGPT helped synthesize dates here """
        time_index = pd.date_range(
            datetime(year, 1, 1), datetime(year, 12, 31),
            freq="D"
        )
        
        doy_axis = xr.DataArray(
            np.arange(1, len(time_index) + 1, dtype="int16"),
            coords={ "time":time_index },
            dims=("time",)
        )

        # Broadcast to (time, y, x)
        doy_axis_3d, burn_doy_3d = xr.broadcast(doy_axis, burn_doys)
        burn_daily = (burn_doy_3d == doy_axis_3d).astype("float32")
        burn_daily.name = f_cfg.name

        return burn_daily.to_dataset(name=f_cfg.name)
        """ --------------------------------------------------------------"""

    # ----------------------------------------------------------------------
    # Fetching
    # ----------------------------------------------------------------------
    def _get_hdf_links(self, granules: List[DataGranule]) -> List[str]:
        """ Get unique HDF download link for a list of granules """
        links: list[str] = []
        seen = set()
        for g in granules:
            for u in g["umm"].get("RelatedUrls", []):
                t = u.get("Type","")
                url = str(u.get("URL",""))
                if (url not in seen and
                    t == "GET DATA" and 
                    url.startswith("https") and url.lower().endswith(".hdf")
                ):
                    seen.add(url)
                    links.append(url)
        return links

    def _granule_pattern(self, short_name, tile):
        return f"{short_name}.*.{tile}.*"

    def _fetch_year(self, short_name: str, year: int) -> List[Path]:
        """ fetch data for a given product and year, return Path to existing/fetched .nc file """

        if not short_name or not year:
            return []

        product_folder = MODIS_DIR / f"{short_name}.{year}"
        ex_files = list(product_folder.glob("*.hdf"))

        if ex_files and all([True for f in ex_files if f.name.startswith(f"{short_name}")]):
            print(f"[LAADS] {short_name} Found my yearbook from {year}, DAMN I look good..")
            return ex_files
        
        comb_results = []
        for tile in self.tiles:
            wildcard = self._granule_pattern(short_name, tile)
            granules = earthaccess.search_data(
                short_name=short_name, 
                version=self.version,
                granule_name=wildcard, 
                temporal=(f"{str(year)}-01-01", f"{str(year)}-12-31"),
                day_night_flag="day",
                bounding_box=self.latlon_tup,
                downloadable=True, 
                count=-1,
            )
            if len(granules) > 0:
                comb_results.extend(granules)
        
        if not comb_results:
            print(f"[LAADS] {short_name}: 'We aint found %$#@ for {year}'")
            return list()
        
        unq_links = self._get_hdf_links(comb_results)
        print(f"[LAADS] downloading using {len(unq_links)} links (from {len(granules)} granules) for {year}")

        dl_file_paths = earthaccess.download(
            granules=unq_links, 
            local_path=product_folder, 
            show_progress=True
        )
        return dl_file_paths

    
