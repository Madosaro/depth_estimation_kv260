# src/training/checkpoint.py

import os
import torch

def resolve_history_path(resume_from, history_path: str) -> str:
    if resume_from is None:
        return history_path
    branch_history_path = resume_from.replace(".pth", "_history.json")
    return branch_history_path if os.path.exists(branch_history_path) else history_path

def resolve_checkpoint_path(model_name: str, checkpoints_dir_path: str, resume_epoch: int) -> str:
    candidate_path = os.path.join(checkpoints_dir_path, f"{model_name}_epoch{resume_epoch}.pth")
    if not os.path.exists(candidate_path):
        available = sorted(f for f in os.listdir(checkpoints_dir_path) if f.endswith(".pth"))
        raise FileNotFoundError(
            f"No checkpoint found for epoch {resume_epoch} at '{candidate_path}'. "
            f"Available snapshots in '{checkpoints_dir_path}': {available}"
        )
    return candidate_path


def load_checkpoint_file(load_path: str, device: torch.device):
    if not os.path.exists(load_path):
        print(f"\t|\t|- No existing checkpoint found at '{load_path}'. Training from scratch.")
        return None
    print(f"\t|\t|- Loading checkpoint from '{load_path}'...")
    return torch.load(load_path, map_location=device)


def apply_model_state(model, checkpoint) -> int:
    if checkpoint is None:
        return 0
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"\t|\t|- Model state restored successfully (epoch {start_epoch}).")
    return start_epoch


def apply_training_state(model, optimizer, scheduler, checkpoint) -> int:
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


def build_checkpoint(epoch: int, model, optimizer, scheduler) -> dict:
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }


def save_checkpoint(checkpoint: dict, path: str):
    torch.save(checkpoint, path, _use_new_zipfile_serialization=False)