import os
import glob
import torch
import pandas as pd
import numpy as np


def get_dataframe(dataset_path:str, split:str)->pd.DataFrame:

    path = os.path.join(dataset_path, split)

    rgb_files = sorted(
        glob.glob(os.path.join(path, "**", "rgb_*.png"), recursive=True)
    )

    dep_files = sorted(
        glob.glob(os.path.join(path, "**", "depth_*.png"), recursive=True)
    )
    

    assert len(rgb_files) == len(dep_files), (
       f"Mismatch: {len(rgb_files)} rgb vs {len(dep_files)} depth"
    )
    print(f"Fichiers trouvés : {len(rgb_files)} paires rgb/depth")

    df = pd.DataFrame({"rgb": rgb_files, "depth": dep_files}).sample(
        frac=1, random_state=42
    )

    return df



def predict(img, model, device, min_depth=0.7, max_depth=10.0):
    """
    Predicts a depth map from an input image.
    Accepts a PyTorch tensor [C, H, W] or preprocessed image.
    """
    model.eval()
    
    # 1. Ensure input is a tensor and add batch dimension [1, C, H, W]
    if not isinstance(img, torch.Tensor):
        # Assumes img is a numpy array or PIL image transformed to tensor
        raise TypeError("Input image must be a torch.Tensor. Use torchvision.transforms.ToTensor()(img) first.")
        
    if img.ndim == 3:
        xb = img.unsqueeze(0).to(device)
    else:
        xb = img.to(device) # Already has batch dimension
        
    with torch.no_grad():
        yb = model(xb)
        
    # 2. Bring back to CPU and convert to numpy
    prediction = yb.cpu().squeeze().numpy() # Squeezes (1, 1, H, W) -> (H, W)
    
    # 3. Denormalize back to actual physical depth (meters)
    actual_depth = prediction * (max_depth - min_depth) + min_depth
    
    # Optional: Clip values to protect against model artifacts outside bounds
    actual_depth = np.clip(actual_depth, min_depth, max_depth)
    
    return actual_depth