#utils.py

import os
import glob
import torch
import json

import pandas as pd
import numpy as np

from scipy.ndimage import uniform_filter



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



#--- 1. build model, optimizer, scheduler -----------------------------------
def build_model(model_class, device):
    print(f'\t|- Model :')
    return model_class().to(device)


def build_optimizer(model, lr=0.0001, optimizer_cls=torch.optim.AdamW):
    return optimizer_cls(model.parameters(), lr=lr)


def build_scheduler(optimizer, scheduler_kwargs=None):
    scheduler_kwargs = scheduler_kwargs or dict(mode='min', patience=3, factor=0.5)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)


#--- 2. paths -----------------------------------------------------------------
def get_model_paths(model_name, models_root="./models/"):
    model_dir_path = os.path.join(models_root, model_name)
    checkpoints_dir_path = os.path.join(model_dir_path, "checkpoints")
    history_path = os.path.join(model_dir_path, model_name + "_history.json")
    model_path = os.path.join(model_dir_path, model_name + ".pth")

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(checkpoints_dir_path, exist_ok=True)

    return dict(
        model_dir_path=model_dir_path,
        checkpoints_dir_path=checkpoints_dir_path,
        history_path=history_path,
        model_path=model_path,
    )


def resolve_checkpoint_path(model_name, checkpoints_dir_path, resume_epoch):
    """Resolve a resume_epoch to a specific checkpoint file path. Raises if missing."""
    candidate_path = os.path.join(checkpoints_dir_path, f"{model_name}_epoch{resume_epoch}.pth")
    if not os.path.exists(candidate_path):
        available = sorted(f for f in os.listdir(checkpoints_dir_path) if f.endswith(".pth"))
        raise FileNotFoundError(
            f"No checkpoint found for epoch {resume_epoch} at '{candidate_path}'. "
            f"Available snapshots in '{checkpoints_dir_path}': {available}"
        )
    return candidate_path


#--- 3. checkpoint loading ----------------------------------------------------
def load_checkpoint_file(load_path, device):
    """Just reads the .pth file off disk. Returns None if it doesn't exist."""
    if not os.path.exists(load_path):
        print(f"\t|\t|- No existing checkpoint found at '{load_path}'. Training from scratch.")
        return None
    print(f"\t|\t|- Loading checkpoint from '{load_path}'...")
    return torch.load(load_path, map_location=device)


def apply_model_state(model, checkpoint):
    """Load just the model weights from a checkpoint dict. Used for eval."""
    if checkpoint is None:
        return 0
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"\t|\t|- Model state restored successfully (epoch {start_epoch}).")
    return start_epoch


def apply_training_state(model, optimizer, scheduler, checkpoint):
    """Load model + optimizer + scheduler state. Used for resuming training."""
    if checkpoint is None:
        return 0
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"\t|\t|- Model state restored successfully (epoch {start_epoch}).")
        return start_epoch
    except (KeyError, RuntimeError) as e:
        print(f"\t|\t|- Failed to load checkpoint ({e}). Training from scratch.")
        return 0


#--- 4. history ----------------------------------------------------------------
HISTORY_KEYS = ["loss_train", "loss_val", "MAE_train", "MAE_val",
                "MSE_train", "MSE_val", "Edge_train", "Edge_val",
                "SSIM_train", "SSIM_val", "lr", "epoch_time"]


def empty_history():
    return {k: [] for k in HISTORY_KEYS}


def load_history(history_path):
    """Load a history json if it exists, else return an empty history dict."""
    if not os.path.exists(history_path):
        print(f"\t|\t|- No history file found at '{history_path}', starting fresh.")
        return empty_history()

    print(f"\t|\t|- Loading training history from '{history_path}'.")
    with open(history_path, "r") as f:
        history = json.load(f)
    for key in HISTORY_KEYS:
        history.setdefault(key, [])
    return history


def resolve_history_path(resume_from, history_path):
    """If resuming from a branch checkpoint, prefer its matching history file."""
    if resume_from is None:
        return history_path
    branch_history_path = resume_from.replace(".pth", "_history.json")
    return branch_history_path if os.path.exists(branch_history_path) else history_path


def truncate_history(history, start_epoch):
    history = {k: v[:start_epoch] for k, v in history.items()}
    print(f"\t|\t|- Resuming from epoch {len(history['loss_train'])}")
    return history


def load_for_training(model_class, model_name, device, resume_epoch=None,
                       models_root="./models/", lr=0.0001):
    model = build_model(model_class, device)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = build_scheduler(optimizer)

    paths = get_model_paths(model_name, models_root)

    resume_from = None
    if resume_epoch is not None:
        resume_from = resolve_checkpoint_path(model_name, paths["checkpoints_dir_path"], resume_epoch)

    load_path = resume_from or paths["model_path"]
    checkpoint = load_checkpoint_file(load_path, device)
    start_epoch = apply_training_state(model, optimizer, scheduler, checkpoint)

    history_source = resolve_history_path(resume_from, paths["history_path"])
    history = load_history(history_source)
    if resume_from is not None:
        history = truncate_history(history, start_epoch)

    return model, optimizer, scheduler, start_epoch, history, paths

def load_for_eval(model_class, model_name, device, models_root="./models/"):
    model = build_model(model_class, device)
    paths = get_model_paths(model_name, models_root)

    checkpoint = load_checkpoint_file(paths["model_path"], device)
    apply_model_state(model, checkpoint)
    model.eval()

    history = load_history(paths["history_path"])

    return model, history



def compute_ssim_numpy(img1, img2, data_range=1.0):
    # Constantes de stabilité de la formule SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Moyennes locales (fenêtre par défaut de 7x7)
    ndimage_kwargs = {'size': 7, 'mode': 'reflect'}
    mu1 = uniform_filter(img1, **ndimage_kwargs)
    mu2 = uniform_filter(img2, **ndimage_kwargs)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variances et covariances locales
    sigma1_sq = uniform_filter(img1 ** 2, **ndimage_kwargs) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, **ndimage_kwargs) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, **ndimage_kwargs) - mu1_mu2

    # Formule de la SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # On renvoie la moyenne globale
    return np.mean(ssim_map)