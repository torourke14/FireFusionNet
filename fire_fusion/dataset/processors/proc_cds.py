
import re
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import cdsapi
import torch
import xarray as xr, rioxarray

from .processor import Processor
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import CDS_DIR
from fire_fusion.utils.utils import load_as_xdataset


class ClimateDataStoreService(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        self.DATASET = dataset
        self.BOUNDS = latlon_bounds
        self.max_parallel_req = 3

        CDS_DIR.mkdir(parents=True, exist_ok=True)
        self.client = cdsapi.Client()
        self.dataset_key = "satellite-fire-burned-area"

    def _gen_request_body(self, year: int, months: List[str]):
        # Must process in < 5 year chunks per data limits
        assert len(months) <= 4, f"[CDS] API gets mad with more than 4 at a time"

        if year > 2019:
            return {
                "origin": "esa_cci",
                "sensor": "modis",
                "variable": "all", # "grid_variables",
                "version": "1_1" if year > 2019 else "5_1_1cds",
                "year": [str(year)],
                "area": [49.1, -124.8, 45.4, -117.0],
                # ["2001", "2002", "2003", "2004", "2005", "2006",
                #  "2007", "2008", "2009", "2010", "2011", "2012",
                #  "2013", "2014", "2015", "2016", "2017", "2018", "2019"
                # ],
                "month": months,
                # ["01", "02", "03", "04", "05", "06",
                # "07", "08", "09", "10", "11", "12"],
                "nominal_day": ["01"] 
            }

        else:
            return {

            }


        return {
            "origin": "esa_cci",
            "sensor": "modis",
            "variable": "all", # "grid_variables",
            "version": "1_1" if year > 2019 else "5_1_1cds",
            "year": [str(year)],
            "area": [49.1, -124.8, 45.4, -117.0],
            # ["2001", "2002", "2003", "2004", "2005", "2006",
            #  "2007", "2008", "2009", "2010", "2011", "2012",
            #  "2013", "2014", "2015", "2016", "2017", "2018", "2019"
            # ],
            "month": months,
            # ["01", "02", "03", "04", "05", "06",
            # "07", "08", "09", "10", "11", "12"],
            "nominal_day": ["01"] 
        }
    
    def _zip_filename(self, year) -> Path:
        return CDS_DIR / f"ba_{year}.zip"
    
    def _zip_to_nc(self, zip_path: Path) -> Path:
        extract_dir = zip_path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        nc_file = sorted(extract_dir.glob("*.nc"))
        if len(nc_file) == 0:
            raise RuntimeError(f"No .nc files found in {zip_path}")
        if len(nc_file) > 1:
            raise RuntimeError(f"Expected exactly 1 .nc file in {zip_path}, found {len(nc_file)}: {[p.name for p in nc_file]}")
        return nc_file[0]


    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        feature_by_yr = xr.Dataset()
    
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            requests = {
                executor.submit(self._fetch_ba_year, f_cfg, yr): yr 
                for yr in self.gridref.attrs['years']
            }
            
            for req in as_completed(requests):
                yr = requests[req]
                try:
                    yr_ds = req.result()
                    if yr_ds is None: continue
                    
                    # print("INDICES", yr_ds.indexes)
                    feature_by_yr = xr.merge([ feature_by_yr, yr_ds ], join="outer")

                except Exception as e:
                    print(f"[LAADS] Satellite tried to load data for {yr}, epic fail! --> {e}\n\n")

        # -----------------------------------------------------------------------

        feature_by_yr = feature_by_yr.sortby("time")
        feature_by_yr = self._time_interpolate(feature_by_yr, f_cfg.time_interp)
        feature_by_yr = feature_by_yr.transpose("time", "y", "x")
        return feature_by_yr


    def _fetch(self, year: str, months: List[str]):
        
        

        zip_path = self._zip_filename(year)
        zip_dir = zip_path.with_suffix("")
        request_body = self._gen_request_body(year)

        
        

        existing_nc_file = zip_dir.glob("*.nc")
        if existing_nc_file:
            print(f"[CDS] Year {year}: Found existing .nc")
            return self._nc_to_tensor(existing_nc_file)
        
        if not zip_path.exists():
            self.client.retrieve(self.dataset_key, request).download(str(zip_path))

            nc_file = self._zip_to_nc(zip_path)
        else:
            print(f"[CDS] Year {year}: Found existing .zip")
            
        tensor = self._nc_to_tensor(nc_file)
        print(f"[CDS] Year {year} tensor shape: {tuple(tensor.shape)}")

        return tensor

    def fetch(self, years: List[str]):
        if not years:
            raise ValueError("[CDS] Please call fetch with years")
        
        tensor_by_year: Dict[str, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            requests = {
                executor.submit(self._fetch_year, year): year
                for year in years
            }

            for req in as_completed(requests):
                year = requests[req]
                try:
                    year_tensor = req.result()
                    tensor_by_year[year] = year_tensor
                except Exception as e:
                    print(f"[CDS] [ERROR] Year {year} failed: {e}")
            
        if not tensor_by_year:
            raise RuntimeError("No tensors were created for any year.")

        # Sort by year and ensure shapes match so stacking is valid
        year_order = sorted(tensor_by_year.keys(), key=int)
        shape = tensor_by_year[year_order[0]].shape

        for y in year_order:
            if tensor_by_year[y].shape != shape:
                raise ValueError(f"[CDS] Shape mismatch across years: year {y} has shape {tensor_by_year[y].shape}, expected {shape}")

        # leading year dimension
        merged_tensor = torch.stack([tensor_by_year[y] for y in year_order], dim=0)

        print(f"[CDS] Merged tensor shape: {tuple(merged_tensor.shape)} -- (num_years={len(year_order)}, per_year_shape={shape})")
        return merged_tensor, year_order
    
if __name__ == "__main__":
    dataset = "satellite-land-cover"
    years = [
        "2000", "2001", "2002", "2003", "2004",
        "2005", "2006", "2007", "2008", "2009",
        "2010", "2011", "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020"
    ]
    latlon_bounds = [50, -125, 40, -110]

    cdss = ClimateDataStoreService(dataset, latlon_bounds)
    merged_tensor, year_order = cdss.fetch(years)