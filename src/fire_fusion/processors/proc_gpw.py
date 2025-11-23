import xarray as xr, rioxarray
import numpy as np
import pandas as pd

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import GPW_DIR
from ..dataset.feature_utils import load_as_xarr
from processor import Processor

class GPW(Processor):
    def __init__(self, cfg, master_grid, mCRS):
        super().__init__(cfg, master_grid, mCRS)
        self.srcCRS = "EPSG:4326"
    
    def build_feature(self, f_config: Feature) -> xr.DataArray:
        pop_snapshots: list[xr.DataArray] = []

        for fp in sorted(GPW_DIR.glob("*.tif")):
            year = fp.stem.split["_"][-1] # type: ignore
            print(f"Counting global population count one by one...")

            with load_as_xarr(fp, name=f_config.name) as pop:
                print(f"[GPWv4] loading {fp.parts[-1]}")
                p_grid = pop.where(pop >= 0)

                p_grid = self._preclip_native(p_grid)
                p_grid = self._reproject_to_mgrid(self.gridref, f_config.resampling)

                ts = pd.Timestamp(f"{year}-07-01")
                p_grid_t = p_grid.expand_dims(time=[ts])
                p_grid_t = p_grid_t.expand_dims(time=self.gridref.time_index) # broadcast over time

                pop_snapshots.append(p_grid)
        
        pop_arr = xr.concat(pop_snapshots, dim="time").sortby("time")
        pop_arr = self._time_interpolate(pop_arr, f_config.time_interp)
        pop_arr = pop_arr.transpose("time", "lat", "lon")

        # log1p + z score
        pop_arr = np.log1p(pop_arr)
        pop_arr = (pop_arr - pop_arr.mean()) / (pop_arr.std() + 1e-6)

        pop_arr.name = f_config.name
        return pop_arr
                

