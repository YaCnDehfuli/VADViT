"""Configuration for consolidated regions to image grids."""
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


IMAGE_SIZE = 384
PATCH_SIZE = 16
TAG_MAPPING = {"Vad": 50, "VadS": 80, "VadF": 100}
PROTECTION_MAPPING = {"PAGE_EXECUTE_READWRITE": 45, "PAGE_EXECUTE_WRITECOPY": 0}
ENT_METHOD = "DYNAMIC"
CONSOLIDATED_DIR = env_path("VADVIT_CONSOLIDATED_DIR", "data/BCCC_Consolidated_Dataset")
IMAGE_DATASET_DIR = env_path("VADVIT_IMAGE_DATASETS_DIR", "data/Image_Datasets")
