import re
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import cdsapi
import numpy as np
import xarray


class GridMetProcessor:
    def __init__(
        self
    ):
        return

class ClimateDSService:
    def __init__(
        self,
        dataset = "satellite-land-cover",
        latlon_bounds = [50, -125, 40, -110],
        max_parallel_req = 4,
        variable = "lccs_class",
        out_dir = "./cds/fetched"
    ):
        self.DATASET = dataset
        self.BOUNDS = latlon_bounds
        self.MAX_PARALLEL_REQ = max_parallel_req
        self.VARIABLE = variable
        self.OUT_DIR = Path(out_dir)
        
    def _request_body(self, years: str | List[str]):
        # Must process in < 5 year chunks per data limits
        return {
            "variable": "all",
            "year": [years] if isinstance(years, str) else years,
            "version": [ "v2_0_7cds", "v2_1_1"],
            "area": self.BOUNDS
        }
            
    def _zip_filename(self, year) -> str:
        return self.OUT_DIR / f"landcover_{year}.zip"
    
    def _zip_to_nc(self, zip_path: Path) -> List[Path]:
        """ Convert .zip --> .nc file """
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
    
    def _nc_to_numpy(self, nc_path: Path) -> np.ndarray:
        """ Convert .nc file to torch tensor 
        """
        with xarray.open_dataset(nc_path) as ds:
            if self.VARIABLE not in ds:
                raise KeyError(
                    f"Variable '{self.VARIABLE}' not found in {nc_path}. "
                    f"Available: {list(ds.data_vars.keys())}"
                )
            data_array = ds[self.VARIABLE].load().values  # numpy ndarray
        # return torch.from_numpy(data_array)
        return data_array
    
    def _fetch_year(self, year):
        """ Fetch a single year from CDS
        """
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        zip_filename = self._zip_filename(year)
        zip_dir = zip_filename.with_suffix("")
        request_body = self._request_body(year)

        ex_nc_files = list(zip_dir.glob("*.nc"))
        if ex_nc_files:
            print(f"[CDS] Found existing .nc for {year}")
            return self._nc_to_numpy(ex_nc_files[0])
        
        if not zip_filename.exists():
            client = cdsapi.Client()
            client.retrieve(self.DATASET, request_body).download(str(zip_filename))
            print(f"[CDS] Extracted .nc file for year {year}")
            
        nc_file = self._zip_to_nc(zip_filename)

        tensor = self._nc_to_numpy(nc_file)
        print(f"[CDS] Year {year} tensor shape: {tuple(tensor.shape)}")

        return tensor

    def async_fetch_years(self, years: List[str]):
        nparr_by_year: Dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_REQ) as executor:
            requests = {
                executor.submit(self._fetch_year, year): year
                for year in years
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

        # Sort by year and ensure shapes match so stacking is valid
        year_order = sorted(nparr_by_year.keys(), key=int)
        shape = nparr_by_year[year_order[0]].shape

        for y in year_order:
            if nparr_by_year[y].shape != shape:
                raise ValueError(f"[CDS] Shape mismatch across years: year {y} has shape {nparr_by_year[y].shape}, expected {shape}")

        # leading year dimension
        merged = np.stack([nparr_by_year[y] for y in year_order], dim=0)

        print(f"[CDS] Merged tensor shape: {tuple(merged.shape)} -- (num_years={len(year_order)}, per_year_shape={shape})")
        return merged, year_order
    
    
if __name__ == "__main__":
    years = list([
        "2000", "2001", "2002", "2003", "2004",
        "2005", "2006", "2007", "2008", "2009",
        "2010", "2011", "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020"
    ])
    latlon_bounds = [50, -125, 40, -110]

    cdss = ClimateDSService(latlon_bounds)
    merged_tensor, year_order = cdss.async_fetch_years(years)