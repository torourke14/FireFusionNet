# Who wants to deal with tuples in JSON, anyways??
from rasterio.enums import Resampling
from typing import List, Optional, Tuple, Literal
import numpy as np

InterpType: Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "quintic", "polynomial", "pchip", "barycentric", "krogh", "akima", "makima"]
ResampType: Resampling

class Feature:
    name: str
    # features with same key get processed together
    key: str
    # fill missing pixels in feature's grid
    resampling: Resampling | None
    # clip values before processing
    clip: Tuple[int, int] | None
    # "time" = broadcasting over time D, "existing" = fill missing
    time_interp: Optional[Tuple[Literal["time", "existing"], str]]
    # for rolling means
    agg_time: int | None
    agg_center: bool | False
    # sequence of normalizations
    norms: Optional[List[str]]

    # If feature is used for derived feature, but not as a channel
    is_derivative: bool | False
    is_label: bool | False
    num_classes: int | 1
    one_hot_encode: bool | False
    

def get_config():
    return {
        "LANDFIRE": {
            "data_dir": "./sources/landfire/downloaded",
            "CRS_projection": "EPSG:5070",
            "features": [
                Feature(
                    name = "elevation",
                    key = "elevation",
                    resampling = Resampling.bilinear,
                    clip = (0, 6000),
                    time_interp = ("time", "linear"),
                    norms = ["z_score"]
                ),
                Feature(
                    name = "slope",
                    key = "slope",
                    resampling = Resampling.bilinear,
                    clip = (0, 360),
                    time_interp = "linear",
                    norms = []
                ),
                Feature(
                    name = "aspect",
                    key = "aspect",
                    resampling = Resampling.bilinear,
                    time_interp = ("time", "linear"),
                    norms = ["deg_to_sin"],
                    num_classes = 1,
                    one_hot_encode = False
                ),
                Feature(
                    name = "water_mask",
                    key = "water_mask",
                    resampling = Resampling.nearest,
                    time_interp = ("time", "linear"),
                    num_classes = 1,
                    one_hot_encode = False,
                    is_label=True
                )
            ]
        },
        "NLCD": {
            "data_dir": "./sources/nlcd/downloaded",
            "features": [
                Feature(
                    name = "land_cover",
                    resampling = Resampling.bilinear,
                    clip=(0, 100),
                    time_interp = ("time", "linear"),
                    agg_time = 30,
                    agg_center = True,
                    norms = ["scale_max"],
                    num_classes = 9,
                    one_hot_encode = True
                ),
                Feature(
                    name = "canopy_cover_pct",
                    resampling = Resampling.bilinear,
                    time_interp = ("time", "linear"),
                    agg_time = 30,
                    agg_center = True,
                    norms = ["scale_max"],
                ),
            ]
        },
        "GPW": {
            "data_dir": "./sources/gpw-v4/downloaded",
            "features": [
                Feature(
                    name = "pop_density",
                    resampling = Resampling.bilinear,
                    time_interp = ("time", "linear"),
                    norms = ["log1p"],
                    is_derivative = True
                )
            ]
        },
        "CENSUSROADS": {
            "features": [
                Feature(
                    name = "dist_to_road",
                    resampling = Resampling.nearest,
                    time_interp = ("time", "linear"),
                    norms = ["scale_max"],
                    is_label=False
                ),
            ]
        },
        "MODIS": {
            "data_dir": "./sources/modis/fetched",
            "max_parallel_req": 7,
            # latlon = (lon min, lat_min, lon max, lat max)
            "products": { # short names
                "MOD13Q1": ("061", "250m 16 days NDVI"),   # Vegetation indices
                "MCD64A1": ("061", "Burn Date"),   # Burned area
                "MCD15A2H": ("061", "Lai_500m"),
            },
            # Granule HDF file names are formatted as "<poduct>/<year>/<day-of-year>/<granule-files>.hdf
            # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
            # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
            "tiles": ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"],
            "features": [
                Feature(
                    name = "modis_ndvi_greenness",
                    resampling = Resampling.bilinear,
                    clip=(-0.1, 1.0),
                    time_interp = ("existing", "nearest"),
                    agg_time = 30,
                    agg_center = True,
                    norms = ["z_score"]
                ),
                Feature(
                    name = "modis_burn",
                    resampling = Resampling.bilinear,
                    time_interp = ("existing", "nearest"),
                    norms = ["mask"],
                    is_derivative = True
                ),
                Feature(
                    name = "modis_lai_canopy",
                    resampling = Resampling.bilinear,
                    clip=(0, 10),
                    time_interp = ("existing", "nearest"),
                    agg_time = 60,
                    agg_center = True,
                    norms = ["z_score"],
                ),
            ]
        },
        "GRIDMET": {
            "data_dir": "./sources/gridmet/downloaded",
            "features": [
                Feature(
                    name = "temp_avg",
                    resampling = Resampling.bilinear,
                    clip=(0.1, 1.0),
                    time_interp = ("time", "linear"),
                    agg_time = 60,
                    agg_center = True,
                    norms = ["z_score"],
                ),
                Feature(
                    name = "wind_ew_max",
                    resampling = Resampling.bilinear,
                    clip=(0, 100),
                    time_interp = ("existing", "quadratic"),
                    norms = ["to_sin", "z_score"],
                ),
                Feature(
                    name = "wind_ns_max",
                    resampling = Resampling.bilinear,
                    clip=(0, 100),
                    time_interp = ("existing", "quadratic"),
                    norms = ["to_cos", "z_score"],
                ),
                Feature(
                    name = "precip_1d",
                    resampling = Resampling.bilinear,
                    clip=(0, np.inf),
                    time_interp = ("existing", "linear"),
                    norms = ["z_score"],
                    num_classes = 9,
                    one_hot_encode = True
                ),
                Feature(
                    name = "precip_5d",
                    resampling = Resampling.bilinear,
                    clip=(0, np.inf),
                    time_interp = ("existing", "linear"),
                    agg_time = 5,
                    agg_center = True,
                    norms = ["z_score"],
                    num_classes = 9,
                    one_hot_encode = True
                ),
                Feature(
                    name = "precip_14d",
                    resampling = Resampling.bilinear,
                    clip=(0, np.inf),
                    time_interp = ("existing", "linear"),
                    agg_time = 15,
                    agg_center = True,
                    norms = ["z_score"]
                ),
            ]
        },
        "FIRE_USFS": {
            "data_dir": "./sources/fire_usfs/downloaded",
            "features": [
                # burn = USFS burn | MTBS burn sev > 20% | modis burn
                Feature(
                    name = "burn",
                    time_interp = ("existing", "zero"),
                    norms = ["mask"],
                    is_label = True
                ),
                Feature(
                    name = "burn_area",
                    time_interp = ("existing", "zero"),
                    norms = ["mask"],
                    is_derivative = True,
                    is_label = True
                ),
            ]
        },
        "FIRE_MTBS": {
            "data_dir": "./sources/fire_mtbs/downloaded",
            "features": [
                Feature(
                    name = "burn_severity",
                    time_interp = ("existing", "zero"),
                    norms = ["mask"],
                    is_derivative = True,
                    is_label = True
                ),
            ]
        },
        "DERIVED": {
            "features": [
                Feature(
                    name = "doy_sin"
                ),
                Feature(
                    name = "wildurban_interface",
                    norms = ["log1p", "scale_max"]
                ),
                Feature(
                    name = "doesberg_fw_index",
                    norms = ["z_score"],
                ),
                Feature(
                    # t last mtbs burn | t last modis burn
                    name = "t_last_burn",
                    clip=(0, np.inf),
                    time_interp = ("existing", "nearest"),
                    norms = ["z_score"],
                ),
                Feature(
                    name = "nearby_prev_fire",
                    agg_time = None,
                    agg_center = None,
                    norms = ["mask"],
                    is_label=False
                )
            ]
        },
    }