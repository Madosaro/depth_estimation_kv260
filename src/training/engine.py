# src/training/engine.py

from __future__ import annotations

import torch


def run_epoch(model, dataloader, loss_function, device, w1, w2, w3, w4, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    losses = []
    l1_list, l2_list, l3_list, l4_list = [], [], [], []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)

            if not is_training and torch.isnan(labels - outputs).any():
                continue

            loss, l1, l2, l3, l4 = loss_function(
                outputs, labels, w1, w2, w3, w4, return_components=True
            )

            if is_training:
                loss.backward()
                optimizer.step()

            losses.append(loss.detach().item())
            l1_list.append(l1.detach().item())
            l2_list.append(l2.detach().item())
            l3_list.append(l3.detach().item())
            l4_list.append(l4.detach().item())

    if not losses:
        mode = "training" if is_training else "validation"
        print(f"\t|\t|- Warning: all {mode} batches were skipped (NaNs). Returning NaN losses.")
        return (float('nan'),) * 5

    return (
        sum(losses) / len(losses),
        sum(l1_list) / len(l1_list),
        sum(l2_list) / len(l2_list),
        sum(l3_list) / len(l3_list),
        sum(l4_list) / len(l4_list),
    )