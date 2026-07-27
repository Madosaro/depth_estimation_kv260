# scripts/train.py

import os
import gc
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import load_config
from src.data.dataframe import get_dataframe, split_dataframe
from src.data.transforms import build_transforms
from src.data.dataset import Nyudepth_png
from src.models.unet import UNet
from src.models.setup import load_for_training
from src.losses.mde_loss import MDELoss
from src.training.engine import run_epoch
from src.training.checkpoint import build_checkpoint, save_checkpoint
from src.training.training_report import generate_training_report


def train(cfg, resume_epoch: int = None):
    print("Starting training...\n")

    #--- get and split the dataframe
    print(f'\t|- ', end="")
    dataframe = get_dataframe(cfg.paths.dataset, "train")
    train_df, val_df = split_dataframe(
        dataframe,
        n=cfg.num_imgs,
        split_proportion=cfg.training.split_proportion,
        seed=cfg.seed,
    )

    #--- build transforms and datasets
    shape_t, color_t = build_transforms(train=True)

    train_dataset = Nyudepth_png(
        os.path.join(cfg.paths.dataset, "train"), train_df, shape_t, color_t
    )
    val_dataset = Nyudepth_png(
        os.path.join(cfg.paths.dataset, "train"), val_df, None, None
    )
    print(f'\t|- Datasets : ')
    print(f'\t|\t|- Train : {len(train_dataset)} pairs ({cfg.training.split_proportion*100.0}%)')
    print(f'\t|\t|- Validation : {len(val_dataset)} pairs ({round(1-cfg.training.split_proportion,2)*100}%)')

    #--- device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("\t|- Device : ", device)

    #--- dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )

    #--- model, optimizer, scheduler, history, paths
    model, optimizer, scheduler, start_epoch, history, paths = load_for_training(
        UNet, cfg.model_name, device, resume_epoch,
        vitis_ai_version=cfg.vitis_ai_version, models_root=cfg.paths.models, lr=cfg.training.lr,
    )

    model_path = paths["model"]
    checkpoints_dir_path = paths["checkpoints_dir"]
    history_path = paths["history"]

    #--- loss
    loss_function = MDELoss().to(device)
    w1, w2, w3, w4 = (
        cfg.loss_weights.w1, cfg.loss_weights.w2,
        cfg.loss_weights.w3, cfg.loss_weights.w4,
    )

    print(f'\t|- Epochs : {cfg.epochs}')
    progress_bar = tqdm(range(start_epoch, start_epoch + cfg.epochs), desc="Training", unit="epoch")

    for epoch in progress_bar:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        loss_t, l1_t, l2_t, l3_t, l4_t = run_epoch(
            model, train_dataloader, loss_function, device, w1, w2, w3, w4,
            optimizer=optimizer,
        )
        loss_v, l1_v, l2_v, l3_v, l4_v = run_epoch(
            model, val_dataloader, loss_function, device, w1, w2, w3, w4,
            optimizer=None,
        )

        scheduler.step(loss_v)
        current_lr = optimizer.param_groups[0]['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_duration = time.time() - start_time

        history.log_epoch(
            loss_train=loss_t, loss_val=loss_v,
            MAE_train=l1_t, MAE_val=l1_v,
            MSE_train=l2_t, MSE_val=l2_v,
            Edge_train=l3_t, Edge_val=l3_v,
            SSIM_train=l4_t, SSIM_val=l4_v,
            lr=current_lr, epoch_time=epoch_duration,
        )

        progress_bar.set_postfix(loss=f"{loss_v:.4f}", lr=f"{current_lr:.6f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        checkpoint = build_checkpoint(epoch + 1, model, optimizer, scheduler)
        save_checkpoint(checkpoint, model_path)

        if (epoch + 1) % cfg.checkpoint_every == 0:
            snapshot_path = os.path.join(checkpoints_dir_path, f"{cfg.model_name}_epoch{epoch+1}.pth")
            save_checkpoint(checkpoint, snapshot_path)
            history.save(snapshot_path.replace(".pth", "_history.json"))
        history.save(history_path)

    generate_training_report(history, paths["outputs_dir"])
    
    progress_bar.close()
    print("-" * 40)
    print(f"Training Finished -")
    print(f"\|- saved model at {cfg.model_name}")
    print(f"\|- Training report at {paths['outputs_dir']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume_epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_epoch=args.resume_epoch)