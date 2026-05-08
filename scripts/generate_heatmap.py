import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys
import os

# Add the root folder to Python's path so it can find your other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your model architecture from your train script
from scripts.train import GeoLightningModel, NUM_CLASSES

def generate_attention_map(image_path, checkpoint_path):
    img = Image.open(image_path).convert('RGB')

    # 1. Apply the exact same normalization you used during training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    input_tensor = transform(img).unsqueeze(0) 

    dummy_centroids = torch.zeros((NUM_CLASSES, 2))
    
    # Load your Stage 2 checkpoint
    model = GeoLightningModel(dummy_centroids)
    from transformers import AutoModel
    model.dinov3 = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", attn_implementation="eager")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Extracting DINOv3 attention weights...")
    with torch.no_grad():
        # Force DINOv3 to output its internal attention matrices
        outputs = model.dinov3(input_tensor, output_attentions=True)
        attentions = outputs.attentions[-1] 

    # Isolate the [CLS] token's attention to everything else
    attn = attentions[0, :, 0, 1:].mean(dim=0)

    # DINOv3 uses 4 "Register Tokens" right before the image patches.
    # We must skip them by grabbing ONLY the last 196 values.
    grid_size = 224 // 16  # 14
    num_patches = grid_size * grid_size  # 196
    
    attn = attn[-num_patches:] # Slice off the registers!

    # Reshape the 196 patches back into a 14x14 spatial grid
    attn = attn.reshape(grid_size, grid_size).cpu().numpy()

    # Resize the heatmap back to 224x224 to overlay smoothly on the image
    attn = cv2.resize(attn / attn.max(), (224, 224))

    print("Rendering heatmap...")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax[0].imshow(img.resize((224,224)))
    ax[0].set_title("Original Image Crop", fontweight='bold')
    ax[0].axis('off')

    # Show attention map overlay
    ax[1].imshow(img.resize((224,224)))
    ax[1].imshow(attn, cmap='jet', alpha=0.5) 
    ax[1].set_title("DINOv3 Attention", fontweight='bold')
    ax[1].axis('off')

    plt.tight_layout()
    
    # Save it directly into your new figures folder!
    output_filename = "figures/Attention_Heatmap.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Pick one of the qualitative images you saved earlier!
    TEST_IMAGE = "figures/dyno_tester.jpg" 
    
    # The path to your Stage 2 weights in your new structure
    CHECKPOINT = "checkpoints_stage2/last.ckpt"
    
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Could not find image at {TEST_IMAGE}. Please check the filename!")
    elif not os.path.exists(CHECKPOINT):
        print(f"Error: Could not find checkpoint at {CHECKPOINT}.")
    else:
        generate_attention_map(TEST_IMAGE, CHECKPOINT)