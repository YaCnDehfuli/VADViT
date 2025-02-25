import torch
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm


def generate_transparent_heatmap(attn_map, image_size=(224, 224), colormap=cm.magma):
    """
    Generates a heatmap in RGBA format where alpha (the 4th channel) is set by attn_map.
    Low values become fully transparent; high values become opaque.
    """
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map_resized = cv2.resize(attn_map, image_size, interpolation=cv2.INTER_CUBIC)
    heatmap_rgba = colormap(attn_map_resized)  
    heatmap_rgba[..., 3] = attn_map_resized  

    return heatmap_rgba 

def overlay_attention(image, cls_attention):

    if isinstance(image, torch.Tensor):
        image = image.squeeze().detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] < 5:
            image = np.transpose(image, (1, 2, 0))  

    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    attn_map = cls_attention.squeeze().detach().cpu().numpy()
    attn_map = attn_map.reshape(7, 7)

    heatmap_rgba = generate_transparent_heatmap(
        attn_map,
        image_size=(image.shape[1], image.shape[0])
    )

    image_float = image.astype(np.float32) / 255.0  

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_float)                  
    ax.imshow(heatmap_rgba, interpolation='nearest')  
    ax.axis("off")
    ax.set_title("Original Image with Transparent Heatmap Overlay")
    plt.show()

    
