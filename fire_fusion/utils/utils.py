import os
from pathlib import Path
import math
import random
from typing import Dict, List
import warnings

import numpy as np
import torch
import torch.optim as optim
import xarray as xr, rioxarray
import rasterio
from rasterio.errors import NotGeoreferencedWarning

BASE_DIR = Path(__file__).resolve().parent



def K_to_F(k):
    return (k * (9/5)) - 459.67


def F_to_K(f):
    return (f + 459.67) * 5/9


def read_hdf_file(subdataset: str, var = None):
    data = rioxarray.open_rasterio(subdataset, variable=var, masked=True)

    if "band" in data.dims and data.sizes.get("band", 1) == 1:  # type: ignore
        data = data.squeeze("band")  # type: ignore
    data.name = np.var # type: ignore
    return data


def load_as_xdataset(
    file: Path,
    variables: List[str] = [],
) -> xr.Dataset:
    suffix = file.suffix.lower()
    try:
        if suffix == ".hdf":
            if not variables:
                raise ValueError("[LOAD_AS_XDATASET] For .hdf need variables list")

            v_dict: Dict[str, xr.DataArray] = {}

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(file) as src:
                    sds_list = src.subdatasets

            if not sds_list:
                raise ValueError(f"[LOAD_AS_XDATASET] No subdatasets for {file.name}. run 'conda install libgdal-hdf4' and try again?")

            for var in variables:
                try:
                    sub = next(sd for sd in sds_list if sd.endswith(f":{var}"))
                except StopIteration:
                    raise ValueError(f"No subdataset ending with ':{var}' in {file.name}. Available:\n{sds_list}")

                da = rioxarray.open_rasterio(sub, masked=True)
                if "band" in da.dims and da.sizes.get("band", 1) == 1: # type: ignore
                    da = da.squeeze("band") # type: ignore
                da.name = var # type: ignore
                v_dict[var] = da # type: ignore

            return xr.Dataset(v_dict)
        
        print("[LOAD_AS_XDATASET] Dont know this file type")
        return xr.Dataset()

    except Exception as e:
        print(f"[LOAD_AS_XDATASET] Failed to load file '{file.stem}':", e)
        return xr.Dataset()



def load_as_xarr(
    file: Path,
    name: str,
    variable = None, # 
    grid = None,
    no_data_val = None,
) -> xr.DataArray:
    """
    load file with a specific variable. To ensure DataArray return type, must provide variable
    """
    suffix = file.suffix.lower()

    try:
        # Landfire, NLCD, GPWv4
        if suffix in {".tif", ".tiff"}:
            darr = rioxarray.open_rasterio(file, masked=True)

            if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
                darr = darr.squeeze("band") # type: ignore

        # gridMET, ESA_CCI, 
        elif suffix == ".nc":
            ds = xr.open_dataset(file, 
                engine="netcdf4", 
                decode_coords="all",
                decode_times=True
            )
            if variable is None:
                raise ValueError(f"[LOAD_AS_XARR] expects variable. Options are: {list(ds.data_vars.keys())}")
            if variable not in ds:
                raise KeyError(f"{variable} not in dataset. Available: {list(ds.data_vars.keys())}")
            darr = ds[variable]
            del ds

        # LAADS
        elif suffix == ".hdf":
            if grid is None or variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable and grid for .hdf files")
            
            with rasterio.open(file) as src:
                sds = src.subdatasets
                if not sds: raise ValueError("No hdf5 subdatasets")

            for sd in sds:
                if sd.endswith(f":{variable}"):
                    darr = read_hdf_file(sd, variable)

        # None yet?
        elif suffix in {".h5", ".hdf5"}:
            ds = xr.open_dataset(file, 
                engine="h5netcdf", 
                decode_coords="all", 
                decode_times=True
            )

            if variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable for .h5/.hdf files")
            if variable == "all":
                return ds[:, :] # return
            if variable not in ds:
                raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
            
            darr = ds[variable]
            del ds

        if no_data_val is not None:
            darr = darr.where(darr != no_data_val, other=np.nan) # type: ignore

        darr.name = name # type: ignore
        return darr # type: ignore

    except Exception as e:
        print(f"[LOAD_AS_XARR] Failed to load file '{file.stem}': ", e)
        return xr.DataArray()
    


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """ Naive way to estimate model size """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024



def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device_config(utilization: float = 0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpus = os.cpu_count() or 1
    workers = math.floor(cpus * utilization)
    if torch.cuda.is_available():
        print(f"Device: {device}, {torch.cuda.get_device_name(0)}")
    print(f"Using {workers}/{utilization} CPUs")
    return device, workers



def save_model(model: torch.nn.Module) -> str:
    """ Use this function to save your model in train.py """
    model_name = "wf-risk-model"

    output_path = BASE_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return str(output_path)


class WarmupCosineAnnealingLR:
    """ 
    PyTorch CosineAnnealing learning rate, with a linear warmup step
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    def __init__(self,
        optimizer,
        warmup_steps: int, total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer

        w_steps = max(0, warmup_steps)
        base_lr = float(optimizer.param_groups[0]["lr"])
        start_factor = min_lr / base_lr

        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = start_factor if w_steps > 0 else 1.0,
            total_iters = warmup_steps if warmup_steps > 0 else 1
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = max(1, int(total_steps - warmup_steps)),
            eta_min = min_lr,
        )
        self.sched = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[w_steps]
        )

    def step(self): self.sched.step()
    def state_dict(self): return self.sched.state_dict()
    def load_state_dict(self, sd): self.sched.load_state_dict(sd)
    def get_last_lr(self): return self.sched.get_last_lr()
    @property
    def last_epoch(self): return self.sched.last_epoch