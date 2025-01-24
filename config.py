# config.py
# Training parameters
IMAGE_SIZE = 224
PATCH_SIZE = 32
SEED = 42
MODE = "Multi"
FROZEN_LAYERS = 6
STEPS = 3
BATCH_SIZE = 1
NUM_EPOCHS = 36
LEARNING_RATE = 2e-4

if MODE == "Binary":
    MULTICLASSS = False
    NUM_CLASSES = 2
elif MODE == "Multi":
    MULTICLASSS = True
    NUM_CLASSES = 9
else:
    print("INVALID MODE")

# Dataset path
DATASET_PATH = f"/home/yacn/Datasets/{PATCH_SIZE}_{IMAGE_SIZE}_Datasets/{PATCH_SIZE}_{IMAGE_SIZE}_{MODE}"
AUC_FOLDER = "/home/yacn/AUCs/"
CM_FOLDER = "/home/yacn/CMs/"
SAVE_PATH = f"./models/{MODE}_{PATCH_SIZE}_{IMAGE_SIZE}_{FROZEN_LAYERS}f_{STEPS}u.pt"


EXPERIMENT_NAME = f"{PATCH_SIZE}_{IMAGE_SIZE}_{FROZEN_LAYERS}f_{STEPS}u" 
MODEL_NAME = f"vit_base_patch{PATCH_SIZE}_{IMAGE_SIZE}"
EXPLAINABILITY = True
