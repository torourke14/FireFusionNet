from pathlib import Path
from typing import Optional
from shapely import box
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from src.fire_fusion.config.feature_config import Feature

def K_to_F(k):
    return (k * (9/5)) - 459.67


def F_to_K(f):
    return (f + 459.67) * 5/9


def load_as_xarr(file: Path, name: str,
    no_data_val = None,
    variable = None # for Dataset -> DataArray
) -> xr.DataArray:
    suffix = file.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        darr = rioxarray.open_rasterio(file, masked=True)

        if "band" in darr.dims and darr.sizes.get("band", 1) == 1:
            darr = darr.squeeze("band")

    elif suffix == ".nc":
        ds = xr.open_dataset("file.nc")

        if variable is None:
            raise ValueError("For .nc files, 'variable' must be provided to select a DataArray.")
        if variable not in ds:
            raise KeyError(f"{variable} not in dataset. Available: {list(ds.data_vars.keys())}")

    elif suffix in {".h5", ".hdf", ".hdf5"}:
        ds = xr.open_dataset(file, engine="h5netcdf", decode_coords="all")

        if variable is None:
            raise ValueError("For .hdf need to select a variable to convert to dataset")
        if variable == "all":
            return ds[:, :] # return
        if variable not in ds:
            raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
        darr = ds[variable]
        del ds

    if no_data_val is not None:
        darr = darr.where(darr != no_data_val, other=np.nan)

    darr.name = name
    return darr