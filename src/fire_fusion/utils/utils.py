import torch
import torch.optim as optim
import xarray as xr, rioxarray
import numpy as np
import os
import math
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent



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

        if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
            darr = darr.squeeze("band") # type: ignore

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
        darr = darr.where(darr != no_data_val, other=np.nan) # type: ignore

    darr.name = name # type: ignore
    return darr # type: ignore

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