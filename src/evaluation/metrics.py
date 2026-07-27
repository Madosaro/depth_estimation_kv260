# src/evaluation/metrics.py

import os
import time
import torch
import json

import numpy as np

from tqdm import tqdm
from scipy.ndimage import uniform_filter

def compute_ssim_numpy(img1, img2, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ndimage_kwargs = {'size': 7, 'mode': 'reflect'}
    mu1 = uniform_filter(img1, **ndimage_kwargs)
    mu2 = uniform_filter(img2, **ndimage_kwargs)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, **ndimage_kwargs) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, **ndimage_kwargs) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, **ndimage_kwargs) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)

def run_evaluation(model, name, device, dataset, step_name, save_dir="./results"):

    os.makedirs(save_dir, exist_ok=True)

    mae_list, mse_list = [], []
    mae_m_list, mse_m_list = [], []
    edges_list, ssim_list = [], []
    latency_list = []

    per_image_metrics = {}

    total_samples = len(dataset)

    is_cuda = device.type == 'cuda'
    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        _ = model(dummy_input)

    with torch.no_grad():
        for i in tqdm(range(total_samples), desc=f"Evaluating {name[:15]} {step_name[:15]}"):
            img_tensor, y_true_tensor = dataset[i]
            y_true = y_true_tensor.squeeze().cpu().numpy()
            y_true = np.clip(y_true, 0.0, 1.0)

            img_input = img_tensor.unsqueeze(0).to(device)

            if is_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            pred_tensor = model(img_input)

            if is_cuda:
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latency_list.append(latency_ms)

            img_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            y_pred = pred_tensor.squeeze().cpu().numpy()
            y_pred = np.clip(y_pred, 0.0, 1.0)

            y_m_true = y_true * (10.0 - 0.7) + 0.7
            y_m_pred = y_pred * (10.0 - 0.7) + 0.7

            img_mae = float(np.mean(np.abs(y_pred - y_true)))
            img_mae_m = float(np.mean(np.abs(y_m_pred - y_m_true)))
            
            img_mse_raw = np.mean(np.power(y_pred - y_true, 2))
            img_mse_m_raw = np.mean(np.power(y_m_pred - y_m_true, 2))
            
            img_rmse = float(np.sqrt(img_mse_raw))
            img_rmse_m = float(np.sqrt(img_mse_m_raw))

            dy_true = y_true[..., 1:, :] - y_true[..., :-1, :]
            dy_pred = y_pred[..., 1:, :] - y_pred[..., :-1, :]
            dx_true = y_true[..., :, 1:] - y_true[..., :, :-1]
            dx_pred = y_pred[..., :, 1:] - y_pred[..., :, :-1]
            img_edges = float(np.mean(np.abs(dy_pred - dy_true)) + np.mean(np.abs(dx_pred - dx_true)))

            img_ssim = float(compute_ssim_numpy(y_true.squeeze(), y_pred.squeeze(), data_range=1.0))

            mae_list.append(img_mae)
            mae_m_list.append(img_mae_m)
            mse_list.append(img_mse_raw)
            mse_m_list.append(img_mse_m_raw)
            edges_list.append(img_edges)
            ssim_list.append(img_ssim)

  
            try:
                if hasattr(dataset, 'dataframe') and 'rgb' in dataset.dataframe.columns:
                    raw_path = dataset.dataframe['rgb'].iloc[i]
                    img_id = os.path.basename(str(raw_path))
                else:
                    img_id = f"image_{i}"
            except Exception:
                img_id = f"image_{i}"

            per_image_metrics[img_id] = {
                "mae": img_mae,
                "mae_m": img_mae_m,
                "mse": img_rmse,
                "mse_m": img_rmse_m,
                "edges": img_edges,
                "ssim": img_ssim,
                "latency": float(latency_ms),
                "fps": float(img_fps)
            }


    json_filename = os.path.join(save_dir, f"{name}_{step_name}.json")
    with open(json_filename, 'w') as f:
        json.dump(per_image_metrics, f, indent=4)
        
    print(f"Saved per-image metrics to {json_filename}")

    mean_latency = np.mean(latency_list)
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "mae": np.sum(mae_list) / total_samples,
        "mae_m": np.sum(mae_m_list) / total_samples,
        "mse": np.sqrt(np.sum(mse_list) / total_samples),
        "mse_m": np.sqrt(np.sum(mse_m_list) / total_samples),
        "edges": np.sum(edges_list) / total_samples,
        "ssim": np.sum(ssim_list) / total_samples,
        "latency": mean_latency,
        "fps": fps,
    }

def compute_metrics_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    mae_list   = [v.get("mae", np.nan) for v in data.values()]
    mae_m_list = [v.get("mae_m", np.nan) for v in data.values()]
    mse_list   = [v.get("mse", np.nan) for v in data.values()]
    mse_m_list = [v.get("mse_m", np.nan) for v in data.values()]
    edges_list = [v.get("edges", np.nan) for v in data.values()]
    ssim_list  = [v.get("ssim", np.nan) for v in data.values()]
    
    latency_list = [v.get("latency", np.nan) for v in data.values()]
    fps_list     = [v.get("fps", np.nan) for v in data.values()]

    n = len(data)

    def safe_mean(lst):
        valid = [x for x in lst if not np.isnan(x)]
        return np.mean(valid) if valid else float("nan")

    def safe_rmse(lst):
        valid = [x for x in lst if not np.isnan(x)]
        return np.sqrt(np.mean(valid)) if valid else float("nan")

    metrics = {
        "mae": safe_mean(mae_list),
        "mae_m": safe_mean(mae_m_list),
        "mse": safe_rmse(mse_list),
        "mse_m": safe_rmse(mse_m_list),
        "edges": safe_mean(edges_list),
        "ssim": safe_mean(ssim_list),
        "latency": safe_mean(latency_list),
        "fps": safe_mean(fps_list),
    }
    return metrics, n