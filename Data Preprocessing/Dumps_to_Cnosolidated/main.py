"""Extract and consolidate regions from a directory of memory dumps."""
import argparse
import logging
from pathlib import Path

from config import BASE_DIR, CONSOLIDATED_DIR, OUTPUT_DIR
from dump_selector import consolidate_regions
from region_divider import divide_regions_dumps_folder
from region_extractor import extract_regions_dumps_folder

logger = logging.getLogger(__name__)


def process_dataset(base_dir: Path, output_dir: Path, consolidated_dir: Path, resume_from: str | None = None):
    """Process samples, optionally beginning at ``resume_from`` (inclusive)."""
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Input dump directory does not exist: {base_dir}. "
            "Set VADVIT_DUMPS_DIR or pass a valid input directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    started = resume_from is None
    if resume_from is not None:
        candidates = [p.name for family in base_dir.iterdir() if family.is_dir() for p in family.iterdir() if p.is_dir()]
        if resume_from not in candidates:
            raise ValueError(f"Resume sample hash not found below {base_dir}: {resume_from}")

    dataset_dict = {}
    for family_path in sorted((p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        family_folder = family_path.name
        family_dict = {"timeout": 0, "single": 0, "multiple": 0}
        for hash_path in sorted((p for p in family_path.iterdir() if p.is_dir()), key=lambda p: p.name):
            if not started:
                if hash_path.name == resume_from:
                    started = True
                else:
                    continue
            dumps_folder = hash_path / "Dumps"
            if not dumps_folder.is_dir():
                continue
            output_sample_dir = output_dir / family_folder / hash_path.name
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Processing %s/%s", family_folder, hash_path.name)
            flag = extract_regions_dumps_folder(str(dumps_folder), str(output_sample_dir))
            family_dict[flag] += 1
            if flag != "timeout":
                divide_regions_dumps_folder(str(output_sample_dir))
                consolidate_regions(str(output_sample_dir), str(consolidated_dir / family_folder / hash_path.name))
        dataset_dict[family_folder] = family_dict
        logger.info("Overall Dumps Stats: %s", dataset_dict)
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", metavar="HASH", help="resume at this sample hash, inclusive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_dataset(BASE_DIR, OUTPUT_DIR, CONSOLIDATED_DIR, args.resume_from)


if __name__ == "__main__":
    main()
