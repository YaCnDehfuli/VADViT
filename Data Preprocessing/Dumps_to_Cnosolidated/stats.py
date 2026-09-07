"""Report dump-region counts for a generated dataset."""
import argparse
import os
from pathlib import Path

from config import OUTPUT_DIR


def collect_stats(dataset_dir: Path):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}. Set VADVIT_REGIONS_DIR."
        )
    dataset_dict = {}
    for family_path in sorted((p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        counts = {"timeout": 0, "single": 0, "multiple": 0}
        for sample_path in family_path.iterdir():
            if not sample_path.is_dir():
                continue
            count = sum(1 for entry in sample_path.iterdir())
            counts["single" if count == 1 else "multiple" if count > 1 else "timeout"] += 1
        dataset_dict[family_path.name] = counts
    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=OUTPUT_DIR)
    print(collect_stats(parser.parse_args().dataset_dir))
