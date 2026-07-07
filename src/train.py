#train.py

import os
import gc
import torch
import json
import time
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

from src.model import UNet
from src.utils import get_dataframe
from src.dataset import Nyudepth_png
from src.loss import MDELoss


def train_model(model, train_dataloader, optimizer, loss_function, device, w1, w2, w3, w4):
    losses = []
    l1_list, l2_list, l3_list, l4_list = [], [], [], []

    model.train()

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, l1, l2, l3, l4 = loss_function(outputs, labels, w1, w2, w3, w4, True)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())
        l1_list.append(l1.detach().item())
        l2_list.append(l2.detach().item())
        l3_list.append(l3.detach().item())
        l4_list.append(l4.detach().item())

    return (
        sum(losses) / len(losses),
        sum(l1_list) / len(l1_list),
        sum(l2_list) / len(l2_list),
        sum(l3_list) / len(l3_list),
        sum(l4_list) / len(l4_list),
    )


def validate_model(model, val_dataloader, loss_function, device, w1, w2, w3, w4):
    losses = []
    l1_list, l2_list, l3_list, l4_list = [], [], [], []

    model.eval()

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if torch.isnan(labels - outputs).any():
                continue

            loss, l1, l2, l3, l4 = loss_function(outputs, labels, w1, w2, w3, w4, return_components=True)

            losses.append(loss.detach().item())
            l1_list.append(l1.detach().item())
            l2_list.append(l2.detach().item())
            l3_list.append(l3.detach().item())
            l4_list.append(l4.detach().item())

        return (
            sum(losses) / len(losses),
            sum(l1_list) / len(l1_list),
            sum(l2_list) / len(l2_list),
            sum(l3_list) / len(l3_list),
            sum(l4_list) / len(l4_list),
        )


def train(dataset_path: str, epochs: int, num_imgs: int, model_name: str,
          resume_epoch: int = None, checkpoint_every: int = 5):
    """
    Train the MDE UNet model.

    Args:
        dataset_path: path to the dataset root.
        epochs: number of additional epochs to train for this run.
        num_imgs: max number of image pairs to sample from the dataframe.
        model_name: name used for the model directory / files.
        resume_epoch: optional epoch number to resume from. If given, looks for
            "./models/<model_name>/checkpoints/<model_name>_epoch<resume_epoch>.pth".
            Raises FileNotFoundError if that snapshot doesn't exist.
        checkpoint_every: save a permanent, non-overwritten snapshot every N epochs.

    If resume_epoch is not given, resumes from the "latest" checkpoint if it
    exists, else trains from scratch.
    """

    print("Starting training...\n")
    #--- get the dataframe
    print(f'\t|- ', end="")
    dataframe = get_dataframe(dataset_path, "train")

    #--- generate datasets
    n = min(num_imgs, len(dataframe))
    dataframe = dataframe.sample(n=n, replace=False).reset_index(drop=True)

    split_proportion = 0.8
    split_idx = int(len(dataframe) * split_proportion)

    shape_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    color_transform = transforms.Compose([v2.RandomChannelPermutation()])

    train_dataset = Nyudepth_png(
        os.path.join(dataset_path, "train"),
        dataframe[:split_idx].reset_index(drop=True),
        shape_transform,
        color_transform
    )
    val_dataset = Nyudepth_png(
        os.path.join(dataset_path, "train"),
        dataframe[split_idx:].reset_index(drop=True),
        None,
        None
    )
    print(f'\t|- Datasets : ')
    print(f'\t|\t|- Train : {len(train_dataset)} pairs ({split_proportion*100.0}%)')
    print(f'\t|\t|- Validation : {len(val_dataset)} pairs ({round(1-split_proportion,2)*100}%)')

    #--- load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("\t|- Device : ", device)

    #--- set dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )

    #--- load model
    print(f'\t|- Model :')
    model = UNet().to(device)

    #--- create the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    #--- create the model dir and set subfolders/files paths
    model_dir_path = os.path.join("./models/", model_name)
    checkpoints_dir_path = os.path.join(model_dir_path, "checkpoints")
    history_path = os.path.join(model_dir_path, model_name + "_history.json")
    model_path = os.path.join(model_dir_path, model_name + ".pth")  # "latest" checkpoint

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(checkpoints_dir_path, exist_ok=True)

    #--- if resume_epoch is given, resolve it to a snapshot path in the checkpoints dir
    resume_from = None
    if resume_epoch is not None:
        candidate_path = os.path.join(checkpoints_dir_path, f"{model_name}_epoch{resume_epoch}.pth")
        if not os.path.exists(candidate_path):
            available = sorted(
                f for f in os.listdir(checkpoints_dir_path) if f.endswith(".pth")
            )
            raise FileNotFoundError(
                f"No checkpoint found for epoch {resume_epoch} at '{candidate_path}'. "
                f"Available snapshots in '{checkpoints_dir_path}': {available}"
            )
        resume_from = candidate_path

    #--- decide which checkpoint file to load from
    load_path = resume_from if resume_from is not None else model_path

    start_epoch = 0
    if not os.path.exists(load_path):
        print(f"\t|\t|- No existing checkpoint found at '{load_path}'. Training from scratch.")
    else:
        print(f"\t|\t|- Loading checkpoint from '{load_path}'...")
        try:
            checkpoint = torch.load(load_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"\t|\t|- Model state restored successfully (epoch {start_epoch}).")
        except (KeyError, RuntimeError) as e:
            print(f"\t|\t|- Failed to load checkpoint ({e}). Training from scratch.")

    history = {
        "loss_train": [], "loss_val": [],
        "MAE_train": [], "MAE_val": [],
        "RMSE_train": [], "RMSE_val": [],
        "Edge_train": [], "Edge_val": [],
        "SSIM_train": [], "SSIM_val": [],
        "lr": [], "epoch_time": []
    }

    #--- if resuming from a specific branch checkpoint, look for its matching history file
    if resume_from is not None:
        branch_history_path = resume_from.replace(".pth", "_history.json")
        history_source = branch_history_path if os.path.exists(branch_history_path) else history_path
    else:
        history_source = history_path

    if os.path.exists(history_source):
        print(f"\t|\t|- Loading training history from '{history_source}'.")
        with open(history_source, "r") as f:
            history = json.load(f)
        for key in ["loss_train", "loss_val", "MAE_train", "MAE_val", "RMSE_train", "RMSE_val",
                    "Edge_train", "Edge_val", "SSIM_train", "SSIM_val", "epoch_time", "lr"]:
            if key not in history:
                history[key] = []
        #--- if resuming from a branch point earlier than the latest history, truncate the rest
        if resume_from is not None:
            history = {k: v[:start_epoch] for k, v in history.items()}
        print(f"\t|\t|- Resuming from epoch {len(history['loss_train'])}")
    else:
        print(f"\t|\t|- No history file found, starting fresh.")

    #--- loss function declaration and variables initialisation
    loss_function = MDELoss().to(device)
    w1, w2, w3, w4 = 1, 2, 1, 1

    #--- progress bar
    print(f'\t|- Epochs : {epochs}')
    progress_bar = tqdm(range(start_epoch, start_epoch + epochs), desc="Training", unit="epoch")

    for epoch in progress_bar:

        #--- start the "chronometer" to get epochs duration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        #--- start a training session and get training report data
        loss_t, l1_t, l2_t, l3_t, l4_t = train_model(
            model, train_dataloader, optimizer,
            loss_function, device, w1, w2, w3, w4
        )

        #--- switch to evaluation mode and evaluate training perfomance on validation data
        with torch.no_grad():
            loss_v, l1_v, l2_v, l3_v, l4_v = validate_model(
                model, val_dataloader, loss_function, device, w1, w2, w3, w4
            )

        #--- update scheduler
        scheduler.step(loss_v)
        current_lr = optimizer.param_groups[0]['lr']

        #--- Synchronize with GPU and get epoch duration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time

        #--- log training and validation data
        history["loss_train"].append(loss_t)
        history["loss_val"].append(loss_v)
        history["MAE_train"].append(l1_t)
        history["MAE_val"].append(l1_v)
        history["RMSE_train"].append(l2_t)
        history["RMSE_val"].append(l2_v)
        history["Edge_train"].append(l3_t)
        history["Edge_val"].append(l3_v)
        history["SSIM_train"].append(l4_t)
        history["SSIM_val"].append(l4_v)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_duration)

        #--- Update loading bar
        progress_bar.set_postfix(
            loss=f"{loss_v:.4f}",
            lr=f"{current_lr:.6f}"
        )

        #--- clean up cuda
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #--- build checkpoint dict fresh each epoch (fixes reliance on a possibly undefined variable)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        #--- always update the "latest" checkpoint (used for default auto-resume)
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)

        #--- periodically save a versioned snapshot that won't be overwritten
        if (epoch + 1) % checkpoint_every == 0:
            snapshot_path = os.path.join(checkpoints_dir_path, f"{model_name}_epoch{epoch+1}.pth")
            snapshot_history_path = snapshot_path.replace(".pth", "_history.json")
            torch.save(checkpoint, snapshot_path, _use_new_zipfile_serialization=False)
            with open(snapshot_history_path, "w") as f:
                json.dump(history, f, indent=4)
            print(f"\t|\t|- Snapshot saved: '{snapshot_path}'")

        #--- save latest history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

    progress_bar.close()
    print("-"*40)
    print(f"Training Finished - saved model at {model_name}")