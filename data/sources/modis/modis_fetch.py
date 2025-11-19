# https://earthaccess.readthedocs.io/en/stable/
import earthaccess
import torch
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List

from osgeo import gdal
try:
    from pyhdf.SD import SD, SDC
except Exception as e:
    pass

class ModisService:
    def __init__(
        self,
        products,
        tiles,
        latlon_bounds = (-125.0, 40.0, -110.0, 50.0), # yyyy-mm-dd
        out_dir = "./downloads"
    ):
        # Pacific Northwest bounding box (lon_min, lat_min, lon_max, lat_max)
        self.BOUNDING_BOX = latlon_bounds
        
        self.OUT_DIR = Path(out_dir)
        self.OUT_DIR.mkdir(exist_ok=True, parents=True)
        self.auth = earthaccess.login(persist=True)

        self.PRODUCTS = products
        self.TILES = tiles
        
        self.DL_FILE_PATHS: dict[str, List[Path]] = {}
        self.TENSORS: dict[str, torch.Tensor] = {}
        # Map to specific parameter to extract (see JSON)
        self.SAT_PARAM_MAP = {
            "MOD13Q1": "250m 16 days NDVI",
            "MYD11A1": "MOD 1KM L3 LST",
            "MCD15A2H": "MCDPR15A2H",
            "MCD64A1": "MCD64A1",
        }

    def _granule_pattern(self, short_name, tile):
        return f"{short_name}.*.{tile}.*"
    
    def _get_granule_out_dir(self, short_name, version):
        outdir = self.OUT_DIR / f"{short_name}.{version}"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    
    def _export_granule_json(self, granule, short_name, tile):
        export_obj = {
            "meta": granule["meta"],
            "umm": granule["umm"],
            "size": granule["size"],
            "data_links": granule.data_links(in_region=True),
        }

        out_path = Path("json_out") / f"{short_name}_{tile}_sample_granule.json"
        with open(out_path, "w") as f:
            json.dump(export_obj, f, indent=2)

        print(f"  [DEBUG] Exported sample granule JSON → {out_path}")
    
    # -------------------------------------------------------------------
    # Data Extraction
    # -------------------------------------------------------------------

    def _tag_granules(self, granules):
        """ Get unique identifier for the granule """
        lookup = {}
        for g in granules:
            gid = g["meta"]["native-id"]

            if not gid:
                gid = g["umm"].get("GranuleUR", "")
            if not gid:
                gran = g["umm"].get("DataGranule", {})
                ids = gran.get("Identifiers", []) or []
                for ide in ids:
                    if ide.get("IdentifierType") == "ProducerGranuleId":
                        gid = ide.get("Identifier")
                        break
            lookup[gid] = g
        return list(lookup.values())
    
    def _get_granule_hdf_url(self, granule) -> str:
        """ Get the .hdf file from a granule (what we want to download) """
        urls = granule["umm"].get("RelatedUrls", [])

        for u in urls: # Try via direct S3 access first
            if u.get("Type") == "GET DATA VIA DIRECT ACCESS" and u["URL"].endswith(".hdf"):
                return u["URL"]
        for u in urls: # Then try HTTPS
            if u.get("Type") == "GET DATA" and u["URL"].endswith(".hdf"):
                return u["URL"]
        for asset in granule.data_links(access="direct"):
            if asset.endswith(".hdf"):
                return asset
        raise RuntimeError(f"No .hdf file found for granule")
    
    def _parse_modis_filename(self, filename) -> Tuple[int, int]:
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
    
    def fetch_time_range(self, time_range: Tuple[str, str], return_tensors = False, export_json = False):
        if not time_range:
            raise ValueError("[MODIS] Please call fetch with years")
        
        for short_name, version in self.PRODUCTS.items():
            print(f"[MODIS] pulling {short_name}.{version}...")

            comb_results = []
            for tile in self.TILES:
                wildcard = self._granule_pattern(short_name, tile)
                granules = earthaccess.search_data(
                    short_name=short_name,
                    version=version,
                    granule_name=wildcard,
                    temporal=time_range,
                    bounding_box=self.BOUNDING_BOX,
                    downloadable=True,
                    cloud_hosted=True,
                    count=-1

                )
                print(f"[MODIS] Granules for {tile}: {len(granules)}")
                if len(granules) > 0:
                    if export_json:
                        self._export_granule_json(granules[0], short_name, tile)
                    comb_results.extend(granules)

            if not comb_results:
                print(f"[MODIS] No results for {short_name} :(")
                self.DL_FILE_PATHS[short_name] = []
                return self.DL_FILE_PATHS

            unique_granules = self._tag_granules(comb_results)
            print(f"[MODIS] Unique granules for {short_name}: {len(unique_granules)}")

            print(f"[MODIS] Downloading granules...")
            out_dir = self._get_granule_out_dir(short_name, version)
            downloaded_paths = earthaccess.download(
                granules=unique_granules,
                local_path=str(out_dir),
                show_progress=True
            )
            self.DL_FILE_PATHS[short_name] = [Path(p) for p in downloaded_paths]
            print(f"[MODIS] Downloaded {len(downloaded_paths)} files to {out_dir}")

        if return_tensors:
            return self._agg_to_tensors()
        return self.DL_FILE_PATHS

    def _agg_to_tensors(self):
        """
        For each product with downloaded files:
          - read HDFs once
          - extract configured SDS by opering MODIS HDF4 file and extracting SDS into a NumPy array
          - stack into torch.Tensor of shape (N, H, W)
        Returns:
            Dict[str, torch.Tensor] mapping short_name -> tensor.
        """
        def _load_sds_from_hdf(hdf_path: Path, sds_name: str) -> np.ndarray:
            """ Open a MODIS HDF4 file and extract the given SDS as a NumPy array.
            Args:
                hdf_path: path to .hdf file
                sds_name: name of the science dataset inside the HDF
            Returns: 2D NumPy array (usually H x W).
            """
            sd = SD(str(hdf_path), SDC.READ)

            if sds_name not in sd.datasets().keys():
                raise KeyError(
                    f"SDS '{sds_name}' not found in {hdf_path.name}. "
                    f"Available datasets: {list(sd.datasets().keys())}"
                )

            sds = sd.select(sds_name)
            data = sds[:, :]  # full raster
            return data
        
        def _load_sds_from_gdal(hdf_path: Path) -> np.ndarray:
            # Open HDF4 container
            ds = gdal.Open(hdf_path)
            if ds is None:
                raise RuntimeError(f"Could not open {hdf_path}")
            
            # List subdatasets (each MODIS SDS is exposed as a GDAL subdataset)
            sub_ds = ds.GetSubDatasets()

            # subdatasets is a list of (name, description) tuples
            name, _ = sub_ds[0]
            sds_ds = gdal.Open(name)
            band = sds_ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            return arr
    
        tensors: dict[str, torch.Tensor] = {}
        # ./BASE_DIR/{short_name}.{version} -- ./raw_downloads/MOD13QC1.061

        for short_name, version in self.PRODUCTS.items():
            print(f"[MODIS] Building tensor for {short_name}.{version} ===")

            if not self.DL_FILE_PATHS:
                raise RuntimeError(f"[MODIS] No Files stored. Fetch first")
            if short_name not in self.DL_FILE_PATHS:
                raise RuntimeError(f"[MODIS] No downloaded files for {short_name}")

            sds_name = self.SAT_PARAM_MAP.get(short_name)
            if sds_name is None:
                raise ValueError(f"[MODIS] No SDS mapping defined for product '{short_name}'. Update SDS_NAME_MAP.")

            files = self.DL_FILE_PATHS[short_name]
            if not files:
                raise RuntimeError(f"Empty file list for {short_name}")
            
            arrays = []
            print(f"[MODIS] Using SDS: '{sds_name}'")
            print(f"[MODIS] Number of HDF files: {len(files)}")

            for i, hdf_path in enumerate(files, start=1):
                try:
                    arr = _load_sds_from_hdf(hdf_path, sds_name)
                    arrays.append(arr)
                except Exception as e:
                    print(f"[MODIS] [ERROR] Could not read {hdf_path.name} using pyHDF: {e}")
                try:
                    arr = _load_sds_from_gdal(hdf_path)
                    arrays.append(arr)
                except Exception as e:
                    print(f"[MODIS] [ERROR] Could not read {hdf_path.name} using gdal: {e}")
                if i % 50 == 0: print(f"[MODIS] Loaded {i} files...")

            if not arrays:
                raise RuntimeError(f"[MODIS] All HDF reads failed for {short_name}.{version}")

            # Stack along new first axis: (N, H, W)
            np_tensor = np.stack(arrays, axis=0)
            torch_tensor = torch.from_numpy(np_tensor).float()
            print(f"[MODIS] Final tensor shape: {torch_tensor.shape}")

            self.TENSORS[short_name] = torch_tensor
            tensors[short_name] = torch_tensor

        return tensors
if __name__ == "__main__":
    # MODIS short names and versions
    PRODUCTS = {
        "MYD11A1": "061",   # Land Surface Temperature/Emissivity
        "MOD13Q1": "061",   # Vegetation indices
        "MCD15A2H": "061",  # Leaf Area Index (LAI)
        "MCD64A1": "061",   # Burned area
    }

    # Granule file names are formatted as "root/product/year/day-of-year/granule-files.hdf
    # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf
    # equals: MOD13Q1, year 2000, day 49, h09/v05, hash
    # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
    PNW_TILES = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]

    latlon_bounds = (-125.0, 40.0, -110.0, 50.0)

    modis_service = ModisService(
        products=PRODUCTS,
        tiles=PNW_TILES,
        latlon_bounds=latlon_bounds
    )

    modis_service.fetch_time_range(
        time_range=("2000-01-01", "2020-12-31"),
        return_tensors=False,
        export_json=False
    )
    print(modis_service.DL_FILE_PATHS)

    tensors = modis_service._agg_to_tensors()