# https://earthaccess.readthedocs.io/en/stable/
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

import earthaccess
from earthaccess import DataGranule

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import MODIS_DIR
from fire_fusion.utils.utils import load_as_xarr
from processor import Processor

class Modis(Processor):
    def __init__(self, cfg, master_grid, mCRS):
        super().__init__(cfg, master_grid, mCRS)
        self.auth = earthaccess.login(persist=True)

        self.latlon_tup = (self.gridref.lon_min, self.gridref.lat_min, self.gridref.lon_max, self.gridref.lat_max)
        MODIS_DIR.mkdir(exist_ok=True, parents=True)

        # Granule HDF file names are formatted as "<poduct>/<year>/<day-of-year>/<granule-files>.hdf
        # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
        self.tiles = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]
        self.version = "061"
        self.max_parallel_req = 6
        
    
    def build_feature(self, f_config: Feature):
        feat_by_year: list[xr.DataArray] = []
        years = list(self.gridref.time_index.year.unique())
    
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            if f_config.key == "MOD13Q1":
                print("Purchasing satellite to collect more data...")
                requests = {
                    executor.submit(self._fetch_ndvi, f_config, yr): yr 
                    for yr in years
                }
            elif f_config.key == "MCD15A2H":
                print(f"Waiting patiently for satellite to arrive")
                requests = {
                    executor.submit(self._fetch_lai, f_config, yr): yr 
                    for yr in years
                }
            elif f_config.key == "MCD64A1":
                print(f"This thing is awesome!")
                requests = {
                    executor.submit(self._fetch_burns, f_config, yr): yr 
                    for yr in years
                }
            
            for req in as_completed(requests):
                yr = requests[req]
                try:
                    year_data = req.result()
                    feat_by_year.append(year_data)
                except Exception as e:
                    print(f"Satellite broke... heading back to {yr}: {e}")
        
        feat_data = xr.concat(feat_by_year, dim="time").sortby("time")
        feat_data = self._time_interpolate(feat_data, f_config.time_interp)
        feat_data = feat_data.transpose("time", "lat", "lon")
        feat_data.name = f_config.name
        return feat_data
    
    def _parse_date(self, filename):
        """ 
        Parse filename to retrieve year and date
        - ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        """
        year, doy = "", ""
        for p in filename.split("."):
            if (p[0] == "A" and p[1:8].isdigit()):
                year = p[1:5]
                doy = p[5:8]
                break
        if year == "" or doy == "":
            raise ValueError(f"[MODIS] Couldn't parse filename {filename}")
        return datetime(year, 1, 1) + timedelta(days=doy - 1)
    

    def _fetch_ndvi(self, f_cfg: Feature, year: int) -> xr.DataArray:
        nc_files = self._fetch_year(short_name=f_cfg.key or "", year=year)
        year_data = []

        for fp in nc_files:
            with load_as_xarr(file=fp, name=f_cfg.name) as raw:
                arr = self._preclip_native(raw)
                arr = self._reproject_to_mgrid(arr, f_cfg.resampling)

                """ -----------------------------------------------------------------------------------
                    NOTE: BELOW BIT PARSING DONE SOLELY BY CHATGPT; 
                        THESE WERE REALLY DIFFICULT TO FIGURE OUT!!
                ----------------------------------------------------------------------------------- """
                ndvi = arr["250m 16 days NDVI"].astype("float32")
                qa  = arr["250m 16 days VI Quality"].astype("float16")
                
                fill_val = ndvi.attrs.get("_FillValue", -3000)
                ndvi = ndvi.where(ndvi != fill_val)

                # --- QA decoding ---
                ndvi_quality = qa & 0b11                  # bits 0-1
                vi_useful    = (qa >> 2) & 0b1111         # bits 2-5
                adj_cloud    = (qa >> 8) & 0b1            # bit 8
                mixed_cloud  = (qa >> 10) & 0b1           # bit 10
                land_water   = (qa >> 11) & 0b111         # bits 11-13

                good_quality   = ndvi_quality <= 1        # keep 00, 01
                good_useful    = vi_useful < 13           # exclude "not useful" / invalid
                no_adj_cloud   = adj_cloud == 0
                no_mixed_cloud = mixed_cloud == 0
                land_mask      = land_water.isin([1, 2])  # land + coastlines

                ndvi = ndvi.where(good_quality & good_useful & no_adj_cloud & no_mixed_cloud & land_mask)

                scale_factor = float(ndvi.attrs.get("scale_factor", 1.0))
                add_offset   = float(ndvi.attrs.get("add_offset", 0.0))
                ndvi = scale_factor * (ndvi - add_offset)
                """ ------------------------------------------------------------------------------- """

            ts = pd.Timestamp(self._parse_date(fp))
            ndvi = ndvi.expand_dims(time=[ts])
            year_data.append(ndvi)

        return xr.concat(year_data, dim="time").sortby("time")
        
    
    def _fetch_lai(self, f_cfg: Feature, year: int) -> xr.DataArray:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        year_data = []

        for fp in nc_files:
            with load_as_xarr(file=fp, name=f_cfg.name) as raw:
                arr = self._preclip_native(raw)
                arr = self._reproject_to_mgrid(arr, f_cfg.resampling)

                """ -----------------------------------------------------------------------------------
                    NOTE: BELOW BIT PARSING DONE SOLELY BY CHATGPT; 
                        THESE WERE REALLY DIFFICULT TO FIGURE OUT!!
                ----------------------------------------------------------------------------------- """
                lai  = arr["Lai_500m"].astype("float32")
                qc   = arr["FparLai_QC"].astype("uint8")
                qcx  = arr["FparExtra_QC"].astype("uint8")

                # Drop fill + non-terrestrial codes (249–255)
                lai = lai.where((lai >= 0) & (lai <= 100))

                # Primary QC (FparLai_QC)
                modland_good = (qc & 0b1) == 0               # bit 0 == 0
                cloudstate   = (qc >> 3) & 0b11              # bits 3–4
                scf_qc       = (qc >> 5) & 0b111             # bits 5–7
                clouds_ok   = cloudstate.isin([0, 3])        # clear / assumed clear
                retrieval_ok = scf_qc < 4                    # exclude "not produced"

                # Extra QC (FparExtra_QC)
                landsea   = qcx & 0b11                       # bits 0–1
                snow_ice  = (qcx >> 2) & 0b1                 # bit 2
                int_cloud = (qcx >> 5) & 0b1                 # bit 5
                shadow    = (qcx >> 6) & 0b1                 # bit 6
                land_ok   = landsea.isin([0, 1])             # land + shore
                no_snow   = snow_ice == 0
                no_clouds = (int_cloud == 0) & (shadow == 0)

                lai = lai.where(modland_good & clouds_ok & retrieval_ok & land_ok & no_snow & no_clouds)

                # Scale factor and offset
                scale_factor = float(lai.attrs.get("scale_factor", 0.1))
                add_offset   = float(lai.attrs.get("add_offset", 0.0))
                lai = scale_factor * (lai - add_offset)
                """ ------------------------------------------------------------------------------- """

            ts = pd.Timestamp(self._parse_date(fp))
            lai = lai.expand_dims(time=[ts])
            year_data.append(lai)

        return xr.concat(year_data, dim="time").sortby("time")
    

    def _fetch_burns(self, f_cfg: Feature, year: int) -> xr.DataArray:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        year_data = []

        for fp in nc_files:
            with load_as_xarr(file=fp, name=f_cfg.name) as raw:
                arr = self._preclip_native(raw)
                arr = self._reproject_to_mgrid(arr, f_cfg.resampling)

                """ -----------------------------------------------------------------------------------
                    NOTE: BELOW BIT PARSING DONE SOLELY BY CHATGPT; 
                        THESE WERE REALLY DIFFICULT TO FIGURE OUT!!
                ----------------------------------------------------------------------------------- """
                burn = raw["Burn Date"].astype("int16")
                qa   = raw["QA"].astype("uint8")
                
                burn = burn.where(burn >= 0)      # drop -1, -2
                is_land  = (qa & 0b1) == 1        # bit 0
                is_valid = ((qa >> 1) & 0b1) == 1 # bit 1

                burn = burn.where(is_land & is_valid)
                burn_mask = (burn > 0).astype("float32")
                """ ------------------------------------------------------------------------------- """

            ts = pd.Timestamp(self._parse_date(fp))
            burn_mask = burn_mask.expand_dims(time=[ts])
            year_data.append(burn_mask)

        return xr.concat(year_data, dim="time").sortby("time")

    def _granule_pattern(self, short_name, tile):
        return f"{short_name}*.{tile}*.hdf"
          #    f"{short_name}.*.{tile}.*.hdf"
          #    f"{short_name}.*.{tile}.*.hdf"

    def _get_hdf_links(self, granules: List[DataGranule]) -> List[str]:
        """ 
        Get unique HDF download link for a list of granules
        """
        def _get_granule_hdf_urls(granule) -> List[str]:
            # Get the .hdf file link (what we want to download)
            urls = granule["umm"].get("RelatedUrls", [])
            for u in urls:
                if u.get("Type") == "GET DATA VIA DIRECT ACCESS" and u["URL"].endswith(".hdf"):
                    return u["URL"]
            for u in urls:
                if u.get("Type") == "GET DATA" and u["URL"].endswith(".hdf"):
                    return u["URL"]
            return list()
        
        lookup= list()
        for g in granules:
            gid = g["meta"]["native-id"]
            if not gid:
                gid = g["umm"].get("GranuleUR", "")
            if not gid:
                gid = g["umm"]["DataGranule"]["Identifiers"][0].get('Identifier', "")
            if gid:
                lookup.extend(_get_granule_hdf_urls(g))
        return lookup

    def _fetch_year(self, short_name: str, year: int) -> List[Path]:
        """ fetch data for a given product and year
            return Path to existing/fetched .nc file
        """
        product_folder = MODIS_DIR / f"{short_name}.{year}"

        ex_files = list(product_folder.glob("*.nc"))
        if ex_files:
            print(f"Found my yearbook from {year}, DAMN I look good..")
            return ex_files
        
        comb_results = []
        time_range = (f"{str(year)}-01-01", f"{str(year)}-12-31")
        for tile in self.tiles:
            print("")
            wildcard = self._granule_pattern(short_name, tile)
            granules = earthaccess.search_data(
                short_name=short_name, version=self.version,
                granule_name=wildcard, temporal=time_range,
                bounding_box=self.latlon_tup,
                downloadable=True, cloud_hosted=True, count=-1
            )
            if len(granules) > 0:
                comb_results.extend(granules)

        if not comb_results:
            print(f"Told my satellite {short_name} was cooler")
            return list()
        
        print("Adding secret tensors for a fun surprise (JK)")

        dl_file_paths = earthaccess.download(
            granules=self._get_hdf_links(comb_results), 
            local_path=product_folder, 
            show_progress=True
        )
        return dl_file_paths

    
