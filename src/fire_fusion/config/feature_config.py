# Who wants to deal with tuples in JSON, anyways??
from dataclasses import dataclass
from rasterio.enums import Resampling
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from xarray.core.types import InterpOptions
from path_config import CROADS_DIR, USFS_DIR, GPW_DIR, GRIDMET_DIR, LANDFIRE_DIR, MODIS_DIR, NLCD_DIR


CAUSAL_CLASSES = [
    "NATURAL_LIGHTNING",
    "HUMAN",
    "INDUSTRIAL",
    "UNKNOWN"
]

CAUSE_MAP = {
    "NATURAL_LIGHTNING": [
        "1", # 1, 1 - lightning
        "lightning",
        "natural",
        "other natural cause",
    ],
    "HUMAN": [
        "3", # smoking
        "4", # campfire
        "7", # arson
        "8", # children
        # text
        "campfire",
        "camping",
        "arson",
        "incendiary",
        "children",
        "firearms/weapons",
        "human",
        "miscellaneous",
        "other causes",
        "other human cause",
    ],

    "INDUSTRIAL": [
        "2", # equip/vehicle use
        "5", # debris burning
        "6", # railroad
        "9", # misc
        "debris burning",
        "debris/open burning",
        "debris",
        "equip/vehicle use",
        "equipment",
        "equipment use",
        "powgen/trans/distrib",
        "railroad",
        "utilities",
        "vehicle",
    ],

    "UNKNOWN": [
        "0",
        "cause not identified",
        "investigated but und",
        "undetermined",
        "undertermined",
        "",
    ],
}

land_cover_map = {
    0: [11], # water
    1: [12], # snow
    2: [21, 22], # developed, < 49%
    3: [23, 24], # developed >= 50%
    4: [31], # barren
    5: [41, 42, 43], # forest
    6: [52, 71], # farmland
    7: [90, 95], # wetlands
}

@dataclass
class Feature:
    name: str
    is_label: Optional[bool] = False
    is_mask: Optional[bool] = False
    drop: Optional[bool] = False                # data used to derive other features
    key: Optional[str] = ""                     # unique key to access data
    req_param: str = ""
    clip: Optional[Tuple[float, float]] = None
    resampling: Optional[Resampling] = None     # fill missing pixels in feature's grid
    time_interp: Optional[Tuple[str, InterpOptions]] = None # "time" = broadcasting over time D, "existing" = fill missing
    agg_time: Optional[int] = None
    agg_min_period: Optional[int] = None
    agg_center: bool = False
    ds_clip: Optional[Tuple[float, float]] = None   # clip values before processing
    ds_norms: Optional[List[str]] = None        # sequence of normalizations done on constructed feature in the CHANNEL
    num_classes: Optional[int] = 0
    one_hot_encode: Optional[bool] = False
    class_map: Optional[Dict] = {}

def get_f_config():
    return {
        "LANDFIRE": [
            Feature(
                name = "elevation",
                key = "_Elev",
                resampling = Resampling.bilinear,
                clip = (0, 5000),
                time_interp = ("time", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "slope",
                key = "_SlpD",
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "aspect",
                key = "_Asp",
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear"),
            ),
            Feature(
                name = "water_mask",
                key = "_EVC",
                drop = True,
                resampling = Resampling.nearest,
                time_interp = ("time", "linear")
            )
        ],
        "NLCD": [
            Feature(
                name = "lcov_class",
                key = "LndCov",
                resampling = Resampling.nearest,
                time_interp = ("time", "linear"),
                num_classes = 9,
                one_hot_encode = True
            ),
            Feature(
                name = "frac_imp_surface",
                key = "FctImp",
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear"),
                ds_clip = (0, 1),
                ds_norms = ["minmax"]
            ),
            Feature(
                name = "canopy_cover_pct",
                key = "tccconus",
                resampling = Resampling.bilinear,
                ds_clip = (0, 1),
                ds_norms = ["minmax"]
            )
        ],
        "GPW": [
            Feature(
                name = "pop_density",
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear")
            )
        ],
        "MODIS": [
            Feature(
                name = "modis_ndvi",
                key = "MOD13Q1",
                req_param = "250m 16 days NDVI",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "nearest"),
                ds_clip=(-0.1, 1.0),
                ds_norms = ["minmax"]
            ),
            Feature(
                name = "modis_lai_canopy",
                key = "MCD15A2H",
                req_param = "Lai_500m",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "nearest"),
                agg_time = 60,
                agg_center = True,
                ds_clip=(0, 10),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "modis_burn",
                key = "MCD64A1",
                req_param = "Burn Date",
                drop = True,
                time_interp = ("existing", "nearest"),
            ),
        ],
        "GRIDMET": [
            Feature(
                name = "temp_avg",
                key = "tmm",
                clip = (0.0, 120), #Far
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "rhumidity_pct",
                key = "rm",
                resampling = Resampling.bilinear,
                time_interp = ("time", "linear"),
                agg_time = 2,
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "wind_dir",
                key = "th",
                clip = (0, 360),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "quadratic"),
            ),
            Feature(
                name = "wind_mph",
                key = "vs",
                clip = (0, 100),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "quadratic"),
                ds_norms = ["log1p", "z_score"]
            ),
            Feature(
                name = "precip", # inchdes
                key = "pr",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["log1p", "z_score"]
            ),
        ],
        "CENSUSROADS": [
            Feature(
                name = "dist_to_road",
                resampling = Resampling.nearest,
                time_interp = ("time", "linear"),
                ds_norms = ["log1p", "z_score"]
            )
        ],
        "FIRE_USFS": [
            Feature(
                name = "usfs_burn",
                key = "Fire_Occurence",
                drop =  True,
                time_interp = ("existing", "zero")
            ),
            Feature(
                name = "usfs_perimeter",
                key = "Fire_Perimeter",
                drop = True,
                time_interp = ("existing", "zero")
            ),
            Feature(
                name = "usfs_burn_cause",
                drop = True,
                key = "Fire_Cause",
                time_interp = ("existing", "zero")
            ),
        ]
    }

def get_derived_f_config():
    return [
        # labels
        Feature(name = "ignition_tp1", is_label=True),
        Feature(name = "cause_tp1", is_label=True),
        # masks
        Feature(name = "water_mask", is_mask=True),
        Feature(name = "active_burn_mask", is_mask=True),
        Feature(name = "burn_loss_mask", is_mask=True),
        # features
        Feature(
            name = "fire_spatial_roll",
            key = "fire_spatial_roll",
            time_interp = ("existing", "nearest"),
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(
            name = "precip_5d",
            key = "precip_5d",
            resampling = Resampling.bilinear,
            time_interp = ("existing", "linear"),
            agg_center = True,
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(
            name = "precip_14d",
            key = "precip_14d",
            resampling = Resampling.bilinear,
            time_interp = ("existing", "linear"),
            agg_center = True,
            ds_norms = ["log1p", "z_score"]
        ),
        Feature(
            name = "ndvi_anomaly",
            key = "ndvi_anomaly",
            ds_norms = ["z_score"]
        ),
        Feature(
            name = "fosberg_fwi",
            key = 'ffwi',
            ds_norms = ["z_score"],
        ),
        Feature(
            name = "doy_sin",
            ds_norms = ["to_sin"]
        ),
    ]