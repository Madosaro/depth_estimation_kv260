#utils.py

import os
import glob
import torch
import pandas as pd
import numpy as np


def get_dataframe(dataset_path: str, split: str) -> pd.DataFrame:
    """
    Build a shuffled dataframe of matched RGB/depth file pairs for a dataset split.

    Recursively searches `dataset_path/split` for files named `rgb_*.png` and
    `depth_*.png`, pairs them by sorted order, and asserts the counts match.

    Args:
        dataset_path: root path of the dataset.
        split: subfolder name to search (e.g. "train", "val", "test").

    Returns:
        DataFrame with columns "rgb" and "depth" (matched file paths),
        shuffled with a fixed random_state for reproducibility.

    Raises:
        AssertionError: if the number of rgb and depth files differ.
    """
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
    print(f"Files found : {len(rgb_files)} pairs rgb/depth")
    df = pd.DataFrame({"rgb": rgb_files, "depth": dep_files}).sample(
        frac=1, random_state=42
    )
    return df


def predict(
    img: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    min_depth: float = 0.7,
    max_depth: float = 10.0,
) -> np.ndarray:
    """
    Run inference on a single image (or batch) and denormalize the output to metric depth.

    Args:
        img: input tensor, shape (C, H, W) for a single image or (N, C, H, W)
            for a batch. Must already be a torch.Tensor (e.g. via
            torchvision.transforms.ToTensor()).
        model: trained depth model, output expected in normalized [0, 1] range.
        device: device to run inference on.
        min_depth: minimum metric depth (meters) used to denormalize the output.
        max_depth: maximum metric depth (meters) used to denormalize the output.

    Returns:
        Predicted depth map(s) in meters, as a numpy array, clipped to
        [min_depth, max_depth].

    Raises:
        TypeError: if `img` is not a torch.Tensor.
    """
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