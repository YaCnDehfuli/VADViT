# import os
# import shutil

# Categorized_Regioned_Data_Folder = "/home/yacn/Categorized_Regioned_Data"
# consolidated_Regions_dir = "/home/yacn/Consolidated_Categorized_Regions"

# # Ensure the destination directory exists
# os.makedirs(consolidated_Regions_dir, exist_ok=True)

# for family_folder in os.listdir(Categorized_Regioned_Data_Folder):
#     family_path = os.path.join(Categorized_Regioned_Data_Folder, family_folder)
#     if not os.path.isdir(family_path):
#         continue

#     for hash_folder in os.listdir(family_path):
#         hash_path = os.path.join(family_path, hash_folder)
#         if not os.path.isdir(hash_path):
#             continue

#         dumps_folder_path_list = [os.path.join(hash_path, dump) for dump in os.listdir(hash_path)]
#         if not dumps_folder_path_list:
#             continue

#         max_regions = 0
#         chosen_dump_folder_path = dumps_folder_path_list[0]

#         for dump_folder_path in dumps_folder_path_list:
#             dll_path = os.path.join(dump_folder_path, "dlls")
#             exe_path = os.path.join(dump_folder_path, "malware_executable")

#             dll_region_count = len([f for f in os.listdir(dll_path) if f.endswith(".dmp")]) if os.path.exists(dll_path) else 0
#             exe_region_count = len([f for f in os.listdir(exe_path) if f.endswith(".dmp")]) if os.path.exists(exe_path) else 0

#             count = dll_region_count + exe_region_count
#             if count >= max_regions:
#                 max_regions = count
#                 chosen_dump_folder_path = dump_folder_path

#         print(f"Chosen dump folder for {hash_folder}: {chosen_dump_folder_path}")

#         # Create the target folder structure
#         target_dir = os.path.join(consolidated_Regions_dir, family_folder, hash_folder)
#         os.makedirs(target_dir, exist_ok=True)

#         # Move files individually to the target directory
#         for item in os.listdir(chosen_dump_folder_path):
#             source_item_path = os.path.join(chosen_dump_folder_path, item)
#             target_item_path = os.path.join(target_dir, item)

#             if os.path.isfile(source_item_path):
#                 shutil.move(source_item_path, target_item_path)
#             elif os.path.isdir(source_item_path):
#                 shutil.move(source_item_path, target_item_path)




import os
import shutil

def consolidate_regions(parent_dump_folder, target_dir):

    dumps_folder_path_list = [os.path.join(parent_dump_folder, dump) for dump in os.listdir(parent_dump_folder)]
    if len(dumps_folder_path_list) == 0:
        print("skipped")
        return
    max_regions = 0
    chosen_dump_folder_path = dumps_folder_path_list[0]

    for dump_folder_path in dumps_folder_path_list:
        dll_path = os.path.join(dump_folder_path, "dlls")
        exe_path = os.path.join(dump_folder_path, "malware_executable")

        dll_region_count = len([f for f in os.listdir(dll_path) if f.endswith(".dmp")]) if os.path.exists(dll_path) else 0
        exe_region_count = len([f for f in os.listdir(exe_path) if f.endswith(".dmp")]) if os.path.exists(exe_path) else 0

        count = dll_region_count + exe_region_count
        if count >= max_regions:
            max_regions = count
            chosen_dump_folder_path = dump_folder_path

    print(f"Chosen dump folder : {chosen_dump_folder_path.strip("/")[-1]}")

    os.makedirs(target_dir, exist_ok=True)

    # Move files individually to the target directory
    for item in os.listdir(chosen_dump_folder_path):
        source_item_path = os.path.join(chosen_dump_folder_path)
        target_item_path = os.path.join(target_dir)
        shutil.copytree(source_item_path, target_item_path, dirs_exist_ok=True)

        # if os.path.isfile(source_item_path):
        #     shutil.copytree(source_item_path, target_item_path)
        # elif os.path.isdir(source_item_path):
