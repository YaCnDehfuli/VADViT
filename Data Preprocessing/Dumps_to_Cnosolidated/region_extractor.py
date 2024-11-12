import os
import subprocess
from config import *

def process_dump_file(pid, dump_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vadinfo_file = os.path.join(output_dir, "vadinfo.csv")
    vaddump_cmd = ["python3", VOLATILITY, "-f", dump_file, "-o", output_dir , "windows.vadinfo", f"--pid={pid}", "--dump"]

    try:
        with open(vadinfo_file, "w") as outfile:
            subprocess.run(vaddump_cmd, stdout=outfile, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {dump_file} with PID {pid}: {e}")

def extract_regions_dumps_folder(dumps_folder, output_dir):
    dump_files = [f for f in os.listdir(dumps_folder) if f.endswith(".vmem")]

    for i, dump_file in enumerate(sorted(dump_files)):
        dump_path = os.path.join(dumps_folder, dump_file)
        pid = dump_file.split("_")[0]

        current_output_dir = os.path.join(output_dir, f"dump_{i+1}")
        process_dump_file(pid, dump_path, current_output_dir)
