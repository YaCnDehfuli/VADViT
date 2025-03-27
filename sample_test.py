import torch
from torchvision import transforms
from PIL import Image
from config import NUM_CLASSES, SAVE_PATH, MODEL_NAME
from models.ViT_model import ViTForImages
from utils.att_visualization import overlay_attention
from utils.seed import set_seed
import os, re

# --------------- CONFIG ---------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/media/yacn/My Book Duo/Image_Datasets_family/32_224"
family = "HackTool"
hash = "0e1e3867a0fbfd8ac7d65d3ac9d6e9dd1b4778504d92315b22658c6e68d17bf2"
image_path = f"{path}/{family}/{hash}.png"  

# --------------- Load Model ---------------
att_outputs = {}

def get_attention_scores(name):
    def hook(module, input, output):
        qkv = module.qkv(input[0])
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        att_outputs[name] = attn_scores.softmax(dim=-1).detach().cpu()
        return output
    return hook

model = ViTForImages(MODEL_NAME, NUM_CLASSES).to(device)
model.vit.blocks[-1].attn.register_forward_hook(get_attention_scores("attn"))
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

# --------------- Load and Transform Image ---------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# --------------- Predict ---------------
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print(f"Predicted Class: {pred}, Probabilities: {probs.squeeze().cpu().numpy()}")

# --------------- Attention Overlay ---------------
if "attn" in att_outputs:
    cls_attention = att_outputs["attn"][:, 0, 1:].mean(dim=0)  # Class token attention
    overlay_attention(image_tensor[0], cls_attention)


# ---------------- Region Addr to Patch match --------
regions_dir = "/media/yacn/My Book Duo/BCCC_Consolidated_Dataset"
sample_regions_path = f"{regions_dir}/{family}/{hash}"
exe_folder = os.path.join(sample_regions_path, "malware_executable")
dll_folder = os.path.join(sample_regions_path, "dlls")
pattern = re.compile(r'vad\.0x([0-9a-fA-F]+)')

def extract_memory_address(filename):
    match = pattern.search(filename)
    if match:
        return int(match.group(1), 16)
    return None

exe_regs = sorted([exe_reg for exe_reg in os.listdir(exe_folder) if exe_reg.endswith(".dmp")])
dll_regs = sorted([dll_reg for dll_reg in os.listdir(dll_folder) if dll_reg.endswith(".dmp")])

print(f"{len(exe_regs)} EXE REGIONS and  {len(dll_regs)} DLL Regions")
c = 0
for reg in exe_regs:
    print(f"{c}. {reg}")
    c += 1

for reg in dll_regs:
    print(f"{c}. {reg}")
    c += 1

    