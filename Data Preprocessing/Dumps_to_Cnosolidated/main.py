import os
from region_extractor import extract_regions_dumps_folder  # os.path.join(hash_path, "Dumps") as the directory
from region_divider import divide_regions_dumps_folder     # "/home/yacn/DATASET3/Hoax/0a1" as the directory of input
from dump_selector import consolidate_regions              # target dir: (consolidated_Regions_dir, family_folder, hash_folder)
import logging
from config import *

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

dataset_dict = {}
start = 2

for family_folder in os.listdir(BASE_DIR):

    family_dict = {'timeout': 0, 'single': 0, 'multiple': 0}
    family_path = os.path.join(BASE_DIR, family_folder)
    if not os.path.isdir(family_path):
        continue

    sample_count = 0
    for hash_folder in os.listdir(family_path):
        if hash_folder == "8bc53c486cba7fca5ffe4dd43976cbaac6bfb24acc95d23da5ad5bc0e0689a3e":
            start = 0

        if start == 1:
            hash_path = os.path.join(family_path, hash_folder)
            if not os.path.isdir(hash_path):
                continue
            
            dumps_folder = os.path.join(hash_path, "Dumps")
            if os.path.exists(dumps_folder):
                output_sample_dir = os.path.join(OUTPUT_DIR, family_folder, hash_folder)
                os.makedirs(output_sample_dir, exist_ok=True)
                sample_count += 1

                logger.info(f"Processing dump files for: {family_folder}, Sample {sample_count}, Hash {hash_folder} ....")
                flag = extract_regions_dumps_folder(dumps_folder, output_sample_dir)
                family_dict[flag] += 1
                if flag != "timeout":

                    logger.info("Dividing the regions ")
                    divide_regions_dumps_folder(output_sample_dir)

                    logger.info("Selecting the best dump ")
                    consolidate_regions(output_sample_dir, os.path.join(CONSOLIDATED_DIR, family_folder, hash_folder))

                logger.info(f"Completed processing: {family_folder}, Sample {sample_count}, Hash {hash_folder}")
                logger.info(f"\n\n *#*#*#*#*#*#**#*#    Updated {family_folder} Stats: {family_dict}  *#*#*#*#*#*#**#*#  \n")

            dataset_dict[family_folder] = family_dict
            logger.info(f"Overall Dumps Stats: {dataset_dict}")
        
        if start == 0:
            start = 1
