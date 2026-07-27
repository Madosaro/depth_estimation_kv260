#inference.py

import os
import glob
import json
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from random import randint
from PIL import Image
from pynq_dpu import DpuOverlay
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func #pip install scikit-image


#--- pre-processing functions
def pre_process_rgb(rgb_data):
    rgb_data = np.array(
        Image.fromarray(rgb_data).resize((224, 224), resample=Image.BILINEAR),
        dtype=np.float32
    )
    rgb_data = (rgb_data - np.min(rgb_data))/(np.max(rgb_data) - np.min(rgb_data) + 1e-8)
    rgb_data = (rgb_data - 0.5) * 2.0
    return rgb_data

def pre_process_dep(dep_data):
    dep_data = np.array(
        Image.fromarray(dep_data).resize((224, 224), resample=Image.NEAREST),
        dtype=np.float32
    )
    dep_data = (dep_data*10)/(2**16 - 1)
    dep_data = np.clip(dep_data, 0.7, 10.0)
    dep_data = (dep_data - 0.7)/(10.0 - 0.7)
    return dep_data[..., np.newaxis]

#--- processing function
def predict_dpu(img, dpu, input_buffer, output_buffer):
    input_buffer[0][0, ...] = img
    
    # --- Start Hardware Timer ---
    start_time = time.perf_counter()
    job_id = dpu.execute_async(input_buffer, output_buffer)
    dpu.wait(job_id)
    end_time = time.perf_counter()
    # -----------------------------
    
    latency_ms = (end_time - start_time) * 1000.0
    y_pred = output_buffer[0][0, :, :, 0]
    return y_pred, latency_ms

#--- sanity check function
def sanity_check_print(data):
    print(f"|\t|- Format/Shape (Canaux, H, L)  : {data.shape}")
    print(f"|\t|- Data Type                    : {data.dtype}")
    print(f"|\t|- Value Range (Min  Max)       : {data.min().item():.4f} à {data.max().item():.4f}")
    print(f"|\t|- Mean Value                   : {data.mean().item():.4f}")
    print(f"|\t|- Standard Deviation (Std)     : {data.std().item():.4f}")

def plot_image(rgb_data, dep_data_m, image_path="dpu_input_output.png"):
    print(f"|- EXPORT IMAGE: ")
    rgb_display = (rgb_data / 2.0) + 0.5
    rgb_display = np.clip(rgb_display, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im = axes[1].imshow(dep_data_m, cmap="turbo", vmin=0.70, vmax=6.00)
    axes[1].set_title("Quantized Model Prediction")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Distance (m)")

    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"|\t|- Image path : {image_path}")

#--- standalone prediction-only export (one PNG per dataset element, always saved)
# Saved as a raw 16-bit single-channel PNG (no RGB, no matplotlib rendering,
# no colorbar/axes) so the exact depth values survive for further processing
# on the host machine.
# Encoding: pixel value (uint16) = depth_in_meters * 1000  -> depth in millimeters.
# To reload on the host:
#   depth_m = np.array(Image.open(path), dtype=np.float32) / 1000.0
def save_prediction_png(dep_data_m, image_path):
    depth_mm = np.clip(dep_data_m, 0.0, 65.535) * 1000.0
    depth_uint16 = depth_mm.astype(np.uint16)
    Image.fromarray(depth_uint16).save(image_path)

#--- metrics calculation function
def compute_metrics_single_image(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    y_true = np.clip(y_true, 0.0, 1.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    y_m_true = y_true * (10.0 - 0.7) + 0.7
    y_m_pred = y_pred * (10.0 - 0.7) + 0.7

    mae = np.mean(np.abs(y_pred - y_true))
    mae_m = np.mean(np.abs(y_m_pred - y_m_true))
    mse = np.mean(np.power(y_pred - y_true, 2))
    mse_m = np.mean(np.power(y_m_pred - y_m_true, 2))

    dy_true = y_true[1:, :] - y_true[:-1, :]
    dy_pred = y_pred[1:, :] - y_pred[:-1, :]
    dx_true = y_true[:, 1:] - y_true[:, :-1]
    dx_pred = y_pred[:, 1:] - y_pred[:, :-1]
    edges = np.mean(np.abs(dy_pred - dy_true)) + np.mean(np.abs(dx_pred - dx_true))

    ssim_value = ssim_func(y_true, y_pred, data_range=1.0)

    return {
        "mae": mae, "mae_m": mae_m,
        "mse": mse, "mse_m": mse_m,
        "edges": edges, "ssim": ssim_value,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", type=str, default="./CNN_kv260.xmodel", help="Path to the model. Default is CNN_kv260.xmodel")
    ap.add_argument("-d", "--data_path", type=str, default="./img", help="Path to the rgb and depth data files. Default is ./img")
    ap.add_argument("-out", "--output_dir", type=str, default="./output", help="Directory where all outputs (metrics, plots) are saved. Default is ./output")
    ap.add_argument("-o", "--output_file", type=str, default="metrics.json", help="Filename for the metrics output file, saved inside output_dir. Default is metrics.json")
    ap.add_argument("-p", "--plot_results", type=bool, default=False, help="Select wether the program should output the combined RGB+prediction depth map plot. Default is False")
    args = ap.parse_args()

    MODEL = args.model_path
    IMAGE_DIR = args.data_path
    OUTPUT_DIR = args.output_dir
    JSON_PATH = os.path.join(OUTPUT_DIR, args.output_file)
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

    #--- make sure the output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    print("-"*40)
    print("inference.py")
    print("-"*40)

    #--- load overlay
    overlay = DpuOverlay("dpu.bit")
    overlay.load_model(MODEL)
    print(f"|- Model: {MODEL}")

    #--- prepare dpu and I/O tensors
    dpu = overlay.runner

    input_tensors = dpu.get_input_tensors()
    input_tensors_shape = tuple(input_tensors[0].dims)
    output_tensors = dpu.get_output_tensors()
    output_tensors_shape = tuple(output_tensors[0].dims)

    print(f"|- DPU: ")
    print(f"|\t|- inuput : {input_tensors_shape}")
    print(f"|\t|- output : {output_tensors_shape}")

    input_buffer = [np.empty(input_tensors_shape, dtype=np.float32, order="C")]
    output_buffer = [np.empty(output_tensors_shape, dtype=np.float32, order="C")]

    #--- warm-up run to eliminate driver allocation overhead from calculations
    print("|- Warming up DPU hardware pipeline...")
    dummy_input = np.random.randn(*input_tensors_shape).astype(np.float32)
    _, _ = predict_dpu(dummy_input, dpu, input_buffer, output_buffer)

    #--- load data
    print(f"|- Data:")
    rgb_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "rgb_*.png")))
    dep_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "depth_*.png")))
    print(f"|\t|- rgb_file: {len(rgb_files)}")
    print(f"|\t|- dep_filr: {len(dep_files)}")
    print(f"|- Output directory: {OUTPUT_DIR}")
    print(f"|- Predictions PNGs directory: {PREDICTIONS_DIR}")

    all_metrics = {}

    for rgb_path, dep_path in tqdm(list(zip(rgb_files, dep_files)), desc="Running inference"):

        rgb = np.array(Image.open(rgb_path).convert('RGB'))[..., ::-1]
        dep = np.array(Image.open(dep_path), dtype=np.float32)

        #--- preprocess images
        rgb = pre_process_rgb(rgb)
        dep = pre_process_dep(dep)

        #--- process images with precise hardware timing
        dep_pred, latency = predict_dpu(rgb, dpu, input_buffer, output_buffer)
        dep_pred_m = dep_pred * (10.0 - 0.7) + 0.7
        fps = 1000.0 / latency if latency > 0 else 0.0

        metrics = compute_metrics_single_image(dep, dep_pred)

        #--- store metrics + performance latency for this image pair
        key = os.path.basename(rgb_path)
        all_metrics[key] = {k: float(v) for k, v in metrics.items()}
        all_metrics[key]["latency"] = float(latency)
        all_metrics[key]["fps"] = float(fps)

        #--- always export a standalone prediction PNG for every element of the dataset
        pred_name = f"pred_{os.path.splitext(key)[0]}.png"
        pred_path = os.path.join(PREDICTIONS_DIR, pred_name)
        save_prediction_png(np.squeeze(dep_pred_m), image_path=pred_path)

        #--- optionally also export the combined RGB + prediction plot
        if args.plot_results:
            plot_name = f"dpu_input_output_{os.path.splitext(key)[0]}.png"
            plot_path = os.path.join(OUTPUT_DIR, plot_name)
            plot_image(rgb, np.squeeze(dep_pred_m), image_path=plot_path)

    #--- export metrics (once, after processing all image pairs)
    with open(JSON_PATH, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"|- Metrics export: ")
    print(f"|\t|- Export path: {JSON_PATH}")


if __name__ == "__main__":
    main()