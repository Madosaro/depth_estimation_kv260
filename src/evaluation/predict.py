# /src/metrics/predict

import torch

import numpy as np

def predict(
    img: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    min_depth: float = 0.7,
    max_depth: float = 10.0,
) -> np.ndarray:
    model.eval()
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input image must be a torch.Tensor. Use torchvision.transforms.ToTensor()(img) first.")
    if img.ndim == 3:
        xb = img.unsqueeze(0).to(device)
    else:
        xb = img.to(device)
    with torch.no_grad():
        yb = model(xb)
    prediction = yb.cpu().squeeze().numpy()
    actual_depth = prediction * (max_depth - min_depth) + min_depth
    actual_depth = np.clip(actual_depth, min_depth, max_depth)
    return actual_depth