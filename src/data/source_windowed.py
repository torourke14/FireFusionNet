# https://earthaccess.readthedocs.io/en/stable/
import earthaccess
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr

from processor import Processor
from data.feature_builder import FeatureBuilder
from config import get_config

class Modis(Processor):
    def __init__(self, cfg, master_grid,
        products,
        tiles,
        param_map,
        latlon_bounds = (-125.0, 40.0, -110.0, 50.0), 
        max_parallel_req = 4,
        out_dir = "./downloads",
    ):
        super().__init__(cfg, master_grid)
        # Pacific Northwest bounding box (lon_min, lat_min, lon_max, lat_max)
        self.BOUNDING_BOX = latlon_bounds
        self.MAX_PARALLEL_REQ = max_parallel_req
        self.OUT_DIR = Path(out_dir)
        self.OUT_DIR.mkdir(exist_ok=True, parents=True)
        self.SAT_PARAM_MAP = param_map
        self.auth = earthaccess.login(persist=True)

        self.PRODUCTS = products
        self.TILES = tiles
        self.DL_FILE_PATHS: dict[str, List[Path]] = {}

        self.dbs: Dict[str, FeatureBuilder] = { }
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError
    
    # -------------------------------------------------------------------
    # Granule Utilities
    # -------------------------------------------------------------------
    def _granule_pattern(self, short_name, tile):
        return f"{short_name}*.{tile}*.hdf"
          #    f"{short_name}.*.{tile}.*.hdf"
          #    f"{short_name}.*.{tile}.*.hdf"
    
    def _granule_links_by_id(self, granules):
        """ Get unique HDF download link and store in lookup table by GranuleID """
        def _get_granule_hdf_urls(granule) -> List[str]:
            """ Get the .hdf file link (what we want to download) """
            urls = granule["umm"].get("RelatedUrls", [])
            for u in urls: # Try via direct S3 access first
                if u.get("Type") == "GET DATA VIA DIRECT ACCESS" and u["URL"].endswith(".hdf"):
                    return u["URL"]
            # for u in urls: # Then try HTTPS
            #     if u.get("Type") == "GET DATA" and u["URL"].endswith(".hdf"):
            #         return u["URL"]
            # for asset in granule.data_links(access="direct"):
            #     if asset.endswith(".hdf"):
            #         return asset
            raise RuntimeError(f"No .hdf file found for granule")
        
        lookup = {}
        for g in granules:
            gid = g["meta"]["native-id"]
            if not gid:
                gid = g["umm"].get("GranuleUR", "")
            if not gid:
                gid = g["umm"]["DataGranule"]["Identifiers"][0].get('Identifier', "")
            if gid:
                lookup[gid] = _get_granule_hdf_urls(g)
        return lookup
    
    def _output_sample_json(self, granule, short_name, tile):
        out_path = Path("json_out") / f"{short_name}_{tile}_sample_granule.json"
        with open(out_path, "w") as f:
            json.dump({
                "meta": granule["meta"],
                "umm": granule["umm"],
                "size": granule["size"],
                "data_links": granule.data_links(in_region=True),
            }, f, indent=2)
        print(f"[MODIS] Exported sample granule JSON to {out_path}")

    # -------------------------------------------------------------------
    # Raw value Conversions
    # -------------------------------------------------------------------
    def _mod13q1_scale_ndvi(self, v_int16: np.ndarray) -> np.ndarray:
        v = v_int16.astype("float32")
        v[(v == -3000) | (v < -2000) | (v > 10000)] = np.nan
        # Apply scale factor to [-0.2, 1.0]
        v = v * 0.0001
        # clip
        v = np.clip(v, -0.2, 1.0)
        return self.daily_interpolate(v)
    
    def _mcd64A1_build_burn_area(self, data: xr.Data) -> np.ndarray:


    def _mcd64A1_build_burn_area(self, data: xr.Data) -> np.ndarray:
        
    
    def _mcd15a2h_scale_lai(self, v_uint8: np.ndarray) -> np.ndarray:
        v = v_uint8.astype("float32")
        v[(v == 255) | (v < 0) | (v > 100)] = np.nan

        # scale factor 0.1 -> [0, 10]
        v = v * 0.1
        # most real LAI is about 0–8
        v = np.clip(v, 0.0, 8.0)
        return self.daily_interpolate(v)
    
    def _build_ndvi_anomaly(self, data):
        # group by dayofyear, compute mean/std over all years, then (NDVI − mean_DOY) / std_DOY
        return
    
    def _mcd64a1_encode_burndate(self, v_int16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = v_int16.astype("float32")
        burn_flag = ((v > 0) | (v <= 366)).astype("float32")   # 1 where burned, 0 elsewhere
        # normalized to [-1, 1] for burned pixels
        burn_doy = np.zeros_like(v, dtype="float32")
        valid = (v > 0) | (v <= 366)
        # Scale to (doy - mid_year) / half LEAP year
        burn_doy[valid] = (v[valid] - 183.0) / 183.0
        return burn_flag, burn_doy

    # -------------------------------------------------------------------
    # Data Extraction
    # -------------------------------------------------------------------
    def _hdfs_to_numpy(self, hdf_paths: List[Path]):
        """
        For each product with downloaded files:
          - read HDFs once
          - extract configured SDS by opering MODIS HDF4 file and extracting SDS into a NumPy array
          - stack into torch.Tensor of shape (N, H, W)
        Returns:
            Dict[str, torch.Tensor] mapping short_name -> tensor.
        """
        def _parse_modis_filename(filename) -> Tuple[int, int]:
            """ Parse filename to retrieve year and date
                - ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
            """
            parts = filename.split(".")
            year, doy = "", ""

            for p in parts:
                if (p[0] == "A" and p[1:8].isdigit()):
                    year = p[1:5]
                    doy = p[5:8]
                    break
            if year == "" or doy == "":
                raise ValueError(f"[MODIS] Couldn't parse filename {filename}")
            return year, doy
        
        return

    def _fetch_year(self, short_name, version, year, export_json=False):
        product_dir = self.OUT_DIR / f"{short_name}.{version}.{year}"
        product_dir.mkdir(parents=True, exist_ok=True)

        ex_hdf_files = list(product_dir.glob("*.nc"))
        if ex_hdf_files:
            print(f"[CDS] Found existing .hdf files for {year}")
            return self._hdfs_to_numpy(ex_hdf_files[0])
        else:
            comb_results = []
            time_range = (f"{year}-01-01", f"{year}-12-31")
            for tile in self.TILES:
                wildcard = self._granule_pattern(short_name, tile)
                granules = earthaccess.search_data(
                    short_name=short_name, version=version,
                    granule_name=wildcard, temporal=time_range,
                    bounding_box=self.BOUNDING_BOX,
                    downloadable=True, cloud_hosted=True, count=-1
                )
                if len(granules) > 0:
                    if export_json: self._output_sample_json(granules[0])
                    comb_results.extend(granules)
            if not comb_results:
                print(f"[MODIS] No results for {short_name} :(")
                self.DL_FILE_PATHS[short_name] = []
                return self.DL_FILE_PATHS
        
        granule_links_by_gid = self._granule_links_by_id(comb_results)
        print(f"[MODIS] Downloading {len(granules)} granule files into {str(product_dir)}")

        downloaded_paths = earthaccess.download(
            granules=list(granule_links_by_gid.values()), 
            local_path=str(product_dir), 
            show_progress=True
        )
        
        return self._extract_yearly_hdf_data([Path(p) for p in downloaded_paths])
        
    def async_extract_data(self, short_name: str, version: str, years: List[str],
        export_json=False, load_to_master = False
    ):
        nparr_by_year: Dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_REQ) as executor:
            requests = {
                executor.submit(
                    self._fetch_year, short_name, version, year, export_json
                ): year for year in years
            }
            for req in as_completed(requests):
                year = requests[req]
                try:
                    year_nparr = req.result()
                    nparr_by_year[year] = year_nparr
                except Exception as e:
                    print(f"[CDS] [ERROR] Year {year} failed: {e}")
        if not nparr_by_year:
            raise RuntimeError("No tensors were created for any year.")
        
        year_order = sorted(nparr_by_year.keys(), key=int)
        shape = nparr_by_year[year_order[0]].shape

        # leading year dimension
        merged = np.stack([nparr_by_year[y] for y in year_order], dim=0)

        print(f"[CDS] Merged ndarray shape: {tuple(merged.shape)} -- (num_years={len(year_order)}, per_year_shape={shape})")
        if load_to_master:
            self.add_chunk_to_source_grid(short_name, merged)
        else:
            return merged
        
    def _extract_data(self):
        for product, version in 
