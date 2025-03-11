import argparse
import torch
from models.ViT_model import ViTForImages
from dataset.dataset_loader import ImageDataset
from config import *
from utils.test_utils import test_model
from utils.att_visualization import overlay_attention
from utils.seed import set_seed


att_outputs_dict = {} 
def get_attention_scores(name):
    def hook(module, input, output):
        qkv = module.qkv(input[0])  # Query-Key-Value projection
        q, k, v = qkv.chunk(3, dim=-1)  # Split Q, K, V
        attn_scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # Scaled dot product
        att_outputs_dict[name] = attn_scores.softmax(dim=-1).detach().cpu()
        return output
    return hook


def main():
    set_seed(SEED)
    parser = argparse.ArgumentParser(description="Test ViT model")
    parser.add_argument("--explain", action="store_true", help="Enable attention visualization during testing")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model = ViTForImages(MODEL_NAME, NUM_CLASSES).to(device)
    print(SAVE_PATH)
    model.vit.blocks[-1].attn.register_forward_hook(get_attention_scores('attn'))
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    # Load test dataset
    test_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(len(test_loader))

    # Run evaluation
    test_model(model, test_loader, device, NUM_CLASSES, att_outputs_dict, args.explain)

if __name__ == "__main__":
    main()
