import os
import pandas as pd
import shutil

def read_vad_data(vad_file):
    """Reads the VAD data from the CSV file."""
    return pd.read_csv(vad_file, sep='\t', skiprows=2, header=0, engine='python')

def identify_malware_regions(vad_data, malware_name):
    """Identify the malware executable region."""
    return vad_data[
        (vad_data['File'].str.contains(malware_name, na=False)) & 
        (vad_data['Protection'].str.contains("execute", case=False, na=False))
    ]

def identify_dll_regions(vad_data):
    """Identify DLL regions with execute permissions."""
    return vad_data[
        (vad_data['File'].str.contains(".dll", case=False, na=False)) &
        (vad_data['Protection'].str.contains("execute", case=False, na=False))
    ]

def identify_other_regions(vad_data, malware_df, dll_regions_df):
    """Identify all other regions not in malware or DLL categories."""
    return vad_data[
        ~vad_data.index.isin(malware_df.index) & 
        ~vad_data.index.isin(dll_regions_df.index)
    ]

def split_executable_regions(other_regions_df):
    """Split other regions into those with execute protection and others."""
    remaining_executable_df = other_regions_df[
        other_regions_df['Protection'].str.contains("execute", case=False, na=False)
    ]
    non_executable_remaining_df = other_regions_df[
        ~other_regions_df.index.isin(remaining_executable_df.index)
    ]
    return remaining_executable_df, non_executable_remaining_df

def create_output_folders(dump_directory):
    """Create directories for categorized files."""
    output_folders = {
        "Malware": os.path.join(dump_directory, "malware_executable"),
        "DLLs": os.path.join(dump_directory, "dlls"),
        "Heap_Stack": os.path.join(dump_directory, "heap_and_stack")
    }
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)
    return output_folders

def move_files(category_df, dump_directory, target_folder):
    """Move files to their corresponding folders."""
    for _, row in category_df.iterrows():
        file_name = row['File output']
        source_path = os.path.join(dump_directory, file_name)
        target_path = os.path.join(target_folder, file_name)
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)

def save_csv_files(malware_combined_df, dll_regions_df, non_executable_remaining_df, output_folders):
    """Save category-specific CSV files in their corresponding folders."""
    malware_combined_df.to_csv(os.path.join(output_folders["Malware"], "malware_executable_regions.csv"), index=False)
    dll_regions_df.to_csv(os.path.join(output_folders["DLLs"], "dll_regions.csv"), index=False)
    non_executable_remaining_df.to_csv(os.path.join(output_folders["Heap_Stack"], "heap_and_stack_regions.csv"), index=False)

def process_dump_directory(parent_dumps_folder, subdir, malware_name="malware.exe"):
    """Processes a single dump directory and organizes files based on categories."""
    dump_directory = os.path.join(parent_dumps_folder, subdir)
    vad_file = os.path.join(dump_directory, "vadinfo.csv")
    
    # Read VAD data
    vad_data = read_vad_data(vad_file)
    
    # Identify regions
    malware_df = identify_malware_regions(vad_data, malware_name)
    malware_executable_df = malware_df.iloc[[0]] if not malware_df.empty else pd.DataFrame()
    dll_regions_df = identify_dll_regions(vad_data)
    other_regions_df = identify_other_regions(vad_data, malware_df, dll_regions_df)
    remaining_executable_df, non_executable_remaining_df = split_executable_regions(other_regions_df)

    # Combine malware executable and remaining executable regions
    malware_combined_df = pd.concat([malware_executable_df, remaining_executable_df])

    # Create output folders
    output_folders = create_output_folders(dump_directory)

    # Move files to corresponding folders
    move_files(malware_combined_df, dump_directory, output_folders["Malware"])
    move_files(dll_regions_df, dump_directory, output_folders["DLLs"])
    move_files(non_executable_remaining_df, dump_directory, output_folders["Heap_Stack"])

    # Save categorized CSV files
    save_csv_files(malware_combined_df, dll_regions_df, non_executable_remaining_df, output_folders)

    # print(f"Processing for {subdir} completed successfully. Files and CSVs are organized.")

def divide_regions_dumps_folder(parent_dumps_folder):
    """Processes all dump directories within the parent folder."""
    for subdir in os.listdir(parent_dumps_folder):
        if os.path.isdir(os.path.join(parent_dumps_folder, subdir)):
            process_dump_directory(parent_dumps_folder, subdir)

