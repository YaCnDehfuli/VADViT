import os
import subprocess
import re
from config import *

def is_pid_in_pslist(dump_file, pid):
    pslist_cmd = [
        "python3", VOLATILITY, "-f", dump_file, "windows.pslist"
    ]
    try:
        result = subprocess.run(pslist_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        return bool(re.search(rf"\b{re.escape(f"{pid}")}\b", str(result.stdout)))
    except subprocess.CalledProcessError as e:
        print(f"Error running pslist on {dump_file}: {e}")
        return False

def process_dump_file(pid, dump_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vadinfo_file = os.path.join(output_dir, "vadinfo.csv")
    vaddump_cmd = ["python3", VOLATILITY, "-f", dump_file, "-o", output_dir , "windows.vadinfo", f"--pid={pid}", "--dump"]

    try:
        # logger.info(f"Processing dump file: {dump_file} for PID: {pid}")
        with open(vadinfo_file, "w") as outfile:
            subprocess.run(vaddump_cmd, stdout=outfile, stderr=subprocess.DEVNULL, check=True)
        # logger.info(f"Completed processing of dump file: {dump_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error processing {dump_file} with PID {pid}: {e}")

def extract_regions_dumps_folder(dumps_folder, output_dir):
    # logger.info(f"Processing of dump files of : {dump_file}")
    
    dump_files = [f for f in os.listdir(dumps_folder) if f.endswith(".vmem")]
    flag = None 
    if len(dump_files) == 1:
        flag = 'timeout'
        return flag

    flag = 'multiple'
    for i, dump_file in enumerate(sorted(dump_files)):
        dump_path = os.path.join(dumps_folder, dump_file)
        pid = dump_file.split("_")[0]

        if is_pid_in_pslist(dump_path, pid):
            current_output_dir = os.path.join(output_dir, f"dump_{i+1}")
            process_dump_file(pid, dump_path, current_output_dir)

        else: 
            if i == 1:
                flag = 'single'
            break            

    # logger.info(f" Completed processing of dump files  ")
    return flag

    

# def extract_bccc_regions(base_dir, output_dir):
    
#     hash_path = base_dir

#     dumps_folder = os.path.join(hash_path, "Dumps")
#     if os.path.exists(dumps_folder):
#         output_family_dir = os.path.join(output_dir, "HackTool", "4fd11d7286579a0b5d72786b68ec1b6204a21c45040a440394eac4ad66523937")
#         os.makedirs(output_family_dir, exist_ok=True)
#         extract_regions_dumps_folder(dumps_folder, output_family_dir)
        
    # family_count = 0
    # for family_folder in os.listdir(base_dir):
    #     family_path = os.path.join(base_dir, family_folder)
    #     if not os.path.isdir(family_path):
    #         continue

    #     for hash_folder in os.listdir(family_path):
    #         hash_path = os.path.join(family_path, hash_folder)
    #         if not os.path.isdir(hash_path):
    #             continue

    #         dumps_folder = os.path.join(hash_path, "Dumps")
    #         if os.path.exists(dumps_folder):
    #             output_family_dir = os.path.join(output_dir, family_folder, hash_folder)
    #             os.makedirs(output_family_dir, exist_ok=True)
    #             process_dumps_folder(dumps_folder, output_family_dir)

    #     family_count += 1
    #     logger.info(f"Completed processing family folder: {family_folder} ({family_count} families processed)")

# Main script execution
# if __name__ == "__main__":
#     logger.info("Starting processing of memory dumps")
#     process_bccc_folder(BASE_DIR, OUTPUT_DIR)
#     logger.info("Completed processing of all memory dumps")


