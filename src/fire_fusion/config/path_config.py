# src/fire_fusion/config/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT     = PROJECT_ROOT / "data"
RAW_DATA_DIR  = DATA_ROOT / "raw"
TRAIN_DATA_DIR = DATA_ROOT / "train"
EVAL_DATA_DIR = DATA_ROOT / "eval"
TEST_DATA_DIR = DATA_ROOT / "test"

LANDFIRE_DIR   = RAW_DATA_DIR / "landfire/downloaded"
NLCD_DIR      = RAW_DATA_DIR / "nlcd/downloaded"
GPW_DIR  = RAW_DATA_DIR / "gpw-v4/downloaded"
CROADS_DIR  = RAW_DATA_DIR / "census_roads/downloaded"
USFS_DIR  = RAW_DATA_DIR / "fire_usfs/downloaded"
GRIDMET_DIR   = RAW_DATA_DIR / "gridmet/downloaded"
MODIS_DIR   = RAW_DATA_DIR / "modis/fetched"


