from config import *
import os

dataset_dict = {}
for family in os.listdir("/run/media/yacn/65f70eee-4af1-4d19-b61b-8b23a6eff4d4/home/adam/BCCC_Dataset"):
    family_dict = {'timeout': 0, 'single': 0, 'multiple': 0}
    family_path = os.path.join("/run/media/yacn/65f70eee-4af1-4d19-b61b-8b23a6eff4d4/home/adam/BCCC_Dataset", family)
    if not os.path.isdir(family_path):
        continue

    for hash in os.listdir(family_path):
        hash_path = os.path.join(family_path, hash)
        if not os.path.isdir(hash_path):
            continue
        
        dumps_count = len(os.listdir(hash_path))
        if dumps_count == 1:
            family_dict['single'] += 1
        elif dumps_count > 1:
            family_dict['multiple'] += 1
        else:
            family_dict['timeout'] += 1
            print("NOT possible")
        
    
    dataset_dict[family] = family_dict


print(dataset_dict)