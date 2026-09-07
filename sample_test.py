"""Run single-sample inference and inspect memory regions.

Set VADVIT_IMAGE_DATASETS_DIR and VADVIT_CONSOLIDATED_DIR, or pass the roots
explicitly. A sample family and hash are required; no author-machine sample is
selected implicitly.
"""
import argparse
import os
import re
from pathlib import Path

from config import IMAGE_SIZE, MODEL_NAME, NUM_CLASSES, PATCH_SIZE, SAVE_PATH, env_path

def extract_memory_address(filename):
    match = re.search(r"vad\.0x([0-9a-fA-F]+)", filename)
    return int(match.group(1), 16) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", help="sample family, for example HackTool")
    parser.add_argument("sample_hash", help="sample hash")
    parser.add_argument("--image-root", type=Path, default=env_path("VADVIT_IMAGE_DATASETS_DIR", "data/Image_Datasets"))
    parser.add_argument("--regions-root", type=Path, default=env_path("VADVIT_CONSOLIDATED_DIR", "data/BCCC_Consolidated_Dataset"))
    args = parser.parse_args()
    image_path = args.image_root / f"{PATCH_SIZE}_{IMAGE_SIZE}" / args.family / f"{args.sample_hash}.png"
    sample_regions_path = args.regions_root / args.family / args.sample_hash
    for required in (image_path, sample_regions_path):
        if not required.exists():
            raise FileNotFoundError(f"Required sample input does not exist: {required}")

    import torch
    from PIL import Image
    from torchvision import transforms
    from models.ViT_model import ViTForImages
    from utils.att_visualization import overlay_attention
    from utils.seed import set_seed

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    att_outputs = {}

    def hook(module, input, output):
        qkv = module.qkv(input[0])
        q, k, _ = qkv.chunk(3, dim=-1)
        att_outputs["attn"] = ((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)).softmax(dim=-1).detach().cpu()
        return output

    model = ViTForImages(MODEL_NAME, NUM_CLASSES).to(device)
    model.vit.blocks[-1].attn.register_forward_hook(hook)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(image_tensor), dim=1)
    print(f"Predicted Class: {torch.argmax(probs, dim=1).item()}, Probabilities: {probs.squeeze().cpu().numpy()}")
    if "attn" in att_outputs:
        overlay_attention(image_tensor[0], att_outputs["attn"][:, 0, 1:].mean(dim=0))

    region_index = 0
    for folder_name in ("malware_executable", "dlls"):
        folder = sample_regions_path / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required region folder does not exist: {folder}")
        regions = sorted(p.name for p in folder.iterdir() if p.suffix == ".dmp")
        print(f"{len(regions)} {folder_name} regions")
        for region in regions:
            print(f"{region_index}. {region}")
            region_index += 1


if __name__ == "__main__":
    main()
