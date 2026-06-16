import os
import gc
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

from src.model import UNet
from src.utils import get_dataframe
from src.dataset import Nyudepth_png
from src.loss import MDELoss


def train_model(model, train_dataloader, optimizer, loss_function, device, scheduler, w1, w2, w3, w4):
    losses = []
    rmse_list = []

    model.train()
    
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels, w1, w2, w3, w4)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())

        true_depth = labels * (10.0 - 0.7) + 0.7
        predicted_depth = outputs * (10.0 - 0.7) + 0.7
        rmse = torch.sqrt(torch.mean(torch.pow(predicted_depth - true_depth, 2)))
        rmse_list.append(rmse.detach().item())

        if scheduler is not None:
            scheduler.step(loss.detach())
    
    lr = scheduler.get_last_lr() if scheduler is not None else [0]

    return (
        sum(losses) / len(losses),
        sum(rmse_list)/len(rmse_list),
        lr
    )

def validate_model(model, val_dataloader, loss_function, device, w1, w2, w3, w4):
    losses = []
    rmse_list = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if torch.isnan(labels - outputs).any():
                continue

            loss = loss_function(labels, outputs, w1, w2, w3, w4)
            losses.append(loss.detach().item())

            true_depth = labels * (10.0 - 0.7) + 0.7
            predicted_depth = outputs * (10.0 - 0.7) + 0.7
            rmse = torch.sqrt(torch.mean(torch.pow(predicted_depth - true_depth, 2)))
            rmse_list.append(rmse.detach().item())

        return (
            sum(losses)/len(losses),
            sum(rmse_list)/len(rmse_list)
        )




def train(dataset_path:str, epochs:int, num_imgs:int, model_path:str):
    
    print("Starting training...\n")
    #--- get the dataframe
    dataframe = get_dataframe(dataset_path, "train")

    #--- generate datasets
    n = min(num_imgs, len(dataframe))
    dataframe = dataframe.sample(n=n, replace=False).reset_index(drop=True)

    split_idx = int(len(dataframe) * 0.8)

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
    print(f"Train subset: {len(train_dataset)} | Val subset: {len(val_dataset)}")

    #--- load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("Device : ", device)

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
    model = UNet().to(device)

    if os.path.exists(model_path):
        print(f"Found existing model at '{model_path}'. Loading trained model.")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"No existing model found. Training from scratch.")

    #--- loss function declaration and variables initialisation
    loss_function = MDELoss(device=device)
    loss_train, loss_val = [], []
    rmse_train, rmse_val = [], []
    lr_history = []
    w1, w2, w3, w4 = 1, 1, 1, 1

    #--- optimizer declaration
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    #--- progress bar
    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in progress_bar:
        loss_t, rmse_t, l = train_model(
            model, train_dataloader, optimizer,
            loss_function, device, None, w1, w2, w3, w4
        )
        loss_train.append(loss_t)
        rmse_train.append(rmse_t)
        lr_history.append(l)

        with torch.no_grad():
            loss_v, rmse_v = validate_model(
                model, val_dataloader, loss_function, device, w1, w2, w3, w4
            )
        loss_val.append(loss_v)
        rmse_val.append(rmse_v)

        progress_bar.set_postfix(
            train_loss=f"{loss_t:.4f}",
            train_rmse=f"{rmse_t:.4f}",
            val_loss=f"{loss_v:.4f}",
            val_rmse=f"{rmse_v:.4f}",
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.save(
            model.state_dict(),
            model_path,
            _use_new_zipfile_serialization=False,
        )   

    progress_bar.close()
    print("-"*40)
    print("Training Finished - saved model")
    torch.save(
        model.state_dict(),
        model_path,
        _use_new_zipfile_serialization=False,
    )
