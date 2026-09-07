"""Configuration for dump to consolidated-region preprocessing.

Paths are supplied through environment variables so the pipeline works on any
machine.  The defaults are relative to the repository for convenient local use.
"""
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


BASE_DIR = env_path("VADVIT_DUMPS_DIR", "data/BCCC-Mal-NetMem-2025-Trojan-Onwards")
OUTPUT_DIR = env_path("VADVIT_REGIONS_DIR", "data/BCCC_Dataset")
CONSOLIDATED_DIR = env_path("VADVIT_CONSOLIDATED_DIR", "data/BCCC_Consolidated_Dataset")
VOLATILITY = env_path("VADVIT_VOLATILITY", "volatility3/vol.py")
