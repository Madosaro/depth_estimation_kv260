#train.py

import os
import gc
import torch
import json
import time
import src.utils
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

from src.model import UNet
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

            if not losses:
                print("\t|\t|- Warning: all validation batches were skipped (NaNs). Returning NaN losses.")
                return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        return (
            sum(losses) / len(losses),
            sum(l1_list) / len(l1_list),
            sum(l2_list) / len(l2_list),
            sum(l3_list) / len(l3_list),
            sum(l4_list) / len(l4_list),
        )


def train(dataset_path: str, epochs: int, num_imgs: int, model_name: str,
          resume_epoch: int = None, checkpoint_every: int = 5):
    print("Starting training...\n")
    #--- get the dataframe
    print(f'\t|- ', end="")
    dataframe = src.utils.get_dataframe(dataset_path, "train")

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

    #--- load model and training objects
    model, optimizer, scheduler, start_epoch, history, paths = src.utils.load_for_training(
        UNet, model_name, device, resume_epoch, models_root="./models/", lr=0.0001)

    model_path = paths["model_path"]
    checkpoints_dir_path = paths["checkpoints_dir_path"]
    history_path = paths["history_path"]


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
        history["MSE_train"].append(l2_t)
        history["MSE_val"].append(l2_v)
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