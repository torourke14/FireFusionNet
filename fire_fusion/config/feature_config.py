# Who wants to deal with tuples in JSON, anyways??
from dataclasses import dataclass
from rasterio.enums import Resampling
from typing import Dict, List, Optional, Tuple
from xarray.core.types import InterpOptions
from .path_config import CROADS_DIR, USFS_DIR, GPW_DIR, GRIDMET_DIR, LANDFIRE_DIR, MODIS_DIR, NLCD_DIR


CAUSAL_CLASSES = [
    "NATURAL_LIGHTNING",
    "HUMAN",
    "INDUSTRIAL"
]

CAUSE_RAW_MAP = {
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
        "campfire", "camping",
        "arson", "incendiary", "firearms/weapons",
        "children",
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
        "debris burning", "debris/open burning", "debris",
        "equip/vehicle use", "equipment", "equipment use",
        "powgen/trans/distrib",
        "railroad", "utilities", "vehicle",
    ],
    "UNKNOWN": [
        "0",
        "cause not identified",
        "investigated but und",
        "undetermined", "undertermined",
        "",
    ],
}

LAND_COVER_RAW_MAP = {
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
    name: str = ""
    key: Optional[str] = ""                         # unique key to access data
    clip: Optional[Tuple[float, float]] = None
    resampling: Optional[Resampling] = None         # fill missing pixels in feature's grid
    time_interp: Optional[Tuple[str, InterpOptions]] = None # "time" = broadcasting over time D, "existing" = fill missing
    
    # OHEs
    num_classes: Optional[int] = 0
    one_hot_encode: Optional[bool] = False
    
    # labels and masks
    inputs: Optional[List[str]] = None
    is_label: Optional[bool] = False
    is_mask: Optional[bool] = False
    # derived features
    func: Optional[str] = "" # DerivedProcessor function name
    drop_inputs: Optional[List[str] | None] = None
    ds_clip: Optional[Tuple[float, float]] = None   # clip values after processing
    ds_norms: Optional[List[str]] = None            # sequence of normalizations

def base_feat_config():
    return {
        ### --- Processors -----------------------------------------
        "GRIDMET": [
            Feature(
                name = "temp_avg",
                key = "tmm",
                clip = (0.0, 120.0), #Far
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "rhumidity_pct",
                key = "rm",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "wind_mph",
                key = "vs",
                clip = (0.0, 100.0),
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
            Feature(
                name = "wind_dir",
                key = "th",
                clip = (0.0, 360.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "quadratic"),
            ),
        ],
        "FIRE_USFS": [
            Feature(
                name = "usfs_burn",
                key = "Fire_Occurence",
                time_interp = ("existing", "zero")
            ),
            Feature(
                name = "usfs_burn_cause",
                key = "Fire_Cause",
                time_interp = ("existing", "zero")
            ),
            Feature(
                name = "usfs_perimeter",
                key = "Fire_Perimeter",
                time_interp = ("existing", "zero")
            )  
        ],
        "NLCD": [
            Feature(
                name = "lcov_class",
                key = "LndCov",
                resampling = Resampling.nearest,
                time_interp = ("broadcast", "linear"),
                num_classes = 9,
                one_hot_encode = True
            ),
            Feature(
                name = "frac_imp_surface",
                key = "FctImp",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_clip = (0.0, 1.0),
                ds_norms = ["minmax"]
            ),
            Feature(
                name = "canopy_cover_pct",
                key = "tccconus",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_clip = (0.0, 1.0),
                ds_norms = ["minmax"]
            )
        ],
        "LANDFIRE": [
            Feature(
                name = "elevation",
                key = "_Elev",
                resampling = Resampling.bilinear,
                clip = (0.0, 5000.0),
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "slope",
                key = "_SlpD",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "aspect",
                key = "_Asp",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
            ),
            Feature(
                is_mask=True,
                name = "water_mask",
                key = "_EVC",
                resampling = Resampling.nearest,
                time_interp = ("broadcast", "linear")
            )
        ],
        "MODIS": [
            Feature(
                name = "modis_burn",
                key = "MCD64A1",
                time_interp = ("existing", "nearest"),
            ),
            Feature(
                name = "modis_lai_canopy",
                key = "MCD15A2H",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "nearest"),
                ds_clip=(0.0, 10.0),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "modis_ndvi",
                key = "MOD13Q1",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "nearest"),
            ),
        ],
        "GPW": [
            Feature(
                name = "pop_density",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear")
            )
        ],
        "CENSUSROADS": [
            Feature(
                name = "d_to_road",
                resampling = Resampling.nearest,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["log1p", "z_score"]
            )
        ],
    }


def drv_feat_config():
    """ SEQUENTIAL list of features to derive """
    return [
        Feature(is_label=True,
            name="ign_next",
            func="build_ignition_next",
            inputs=["modis_burn", "usfs_burn", "usfs_perimeter"],
        ),
        Feature(is_mask=True,
            name="act_fire_mask",
            func="build_act_fire_mask",
            inputs=["modis_burn", "usfs_burn", "usfs_perimeter"],
        ),

        Feature(
            name = "fire_spatial_roll",
            func = "build_fire_spatial_rolling",
            time_interp = ("existing", "nearest"),
            inputs=["modis_burn", "usfs_burn", "usfs_perimeter"],
            drop_inputs=["modis_burn", "usfs_burn", "usfs_perimeter"],
            ds_norms = ["log1p", "z_score"],
        ),

        Feature(is_label=True,
            name="ign_next_cause",
            func="build_ign_next_cause",
            inputs=["usfs_burn_cause", "ign_next"],
        ),
        Feature(is_mask=True,
            name="valid_cause_mask",
            func="build_valid_cause_mask",
            inputs=["usfs_burn_cause", "ign_next"], drop_inputs=["usfs_burn_cause"],
        ),
        
        Feature(
            name = "precip_2d",
            func = "build_precip_2d",
            resampling = Resampling.bilinear,
            time_interp = ("existing", "linear"),
            inputs=["precip"],
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(
            name = "precip_5d",
            func = "build_precip_5d",
            resampling = Resampling.bilinear,
            time_interp = ("existing", "linear"),
            inputs=["precip"], drop_inputs=["precip"],
            ds_norms = ["log1p", "z_score"],
        ),
        
        Feature(
            name = "ndvi_anomaly",
            func = "build_ndvi_anomaly",
            inputs=["modis_ndvi"], drop_inputs=["modis_ndvi"],
            ds_clip=(-0.1, 1.0),
            ds_norms = ["z_score"],
        ),
        Feature(
            name = "fosberg_fwi",
            func = 'build_ffwi',
            inputs=["temp_avg", "rhumidity_pct", "wind_mph"],
            ds_norms = ["z_score"],
        ),
        Feature(
            name = "doy_sin",
            func="build_doy_sin",
            inputs=["time"]
        ),
    ]

