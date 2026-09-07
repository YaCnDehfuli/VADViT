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
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def env_path(name, default):
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value

DATASET_PATH = env_path("VADVIT_DATASET_PATH", f"data/{PATCH_SIZE}_{IMAGE_SIZE}_Datasets/{PATCH_SIZE}_{IMAGE_SIZE}_{MODE}")
AUC_FOLDER = env_path("VADVIT_AUC_FOLDER", "outputs/AUCs")
CM_FOLDER = env_path("VADVIT_CM_FOLDER", "outputs/CMs")
SAVE_PATH = env_path("VADVIT_CHECKPOINT", f"models/{MODE}_{PATCH_SIZE}_{IMAGE_SIZE}_{FROZEN_LAYERS}f_{STEPS}u.pt")


EXPERIMENT_NAME = f"{PATCH_SIZE}_{IMAGE_SIZE}_{FROZEN_LAYERS}f_{STEPS}u" 
MODEL_NAME = f"vit_base_patch{PATCH_SIZE}_{IMAGE_SIZE}"
EXPLAINABILITY = True
