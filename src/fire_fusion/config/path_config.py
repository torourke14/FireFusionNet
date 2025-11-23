# src/fire_fusion/config/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT     = PROJECT_ROOT / "data"
RAW_DATA_DIR  = DATA_ROOT / "raw"
TRAIN_DATA_DIR = DATA_ROOT / "train"
EVAL_DATA_DIR = DATA_ROOT / "eval"
TEST_DATA_DIR = DATA_ROOT / "test"

CROADS_DIR  = RAW_DATA_DIR / "census_roads/downloaded"
USFS_DIR  = RAW_DATA_DIR / "fire_usfs/downloaded"
GPW_DIR  = RAW_DATA_DIR / "gpw-v4/downloaded"
GRIDMET_DIR   = RAW_DATA_DIR / "gridmet/downloaded"
LANDFIRE_DIR   = RAW_DATA_DIR / "landfire/downloaded"
MODIS_DIR   = RAW_DATA_DIR / "modis/fetched"
NLCD_DIR      = RAW_DATA_DIR / "nlcd/downloaded"
