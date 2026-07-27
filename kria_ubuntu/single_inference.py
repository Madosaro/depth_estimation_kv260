#single_inference.py

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pynq_dpu import DpuOverlay
from skimage.metrics import structural_similarity as ssim_func  # pip install scikit-image


#--- pre-processing functions
def pre_process_rgb(rgb_data):
    rgb_data = np.array(
        Image.fromarray(rgb_data).resize((224, 224), resample=Image.BILINEAR),
        dtype=np.float32
    )
    rgb_data = (rgb_data - np.min(rgb_data)) / (np.max(rgb_data) - np.min(rgb_data) + 1e-8)
    rgb_data = (rgb_data - 0.5) * 2.0
    return rgb_data


def pre_process_dep(dep_data):
    dep_data = np.array(
        Image.fromarray(dep_data).resize((224, 224), resample=Image.NEAREST),
        dtype=np.float32
    )
    dep_data = (dep_data * 10) / (2**16 - 1)
    dep_data = np.clip(dep_data, 0.7, 10.0)
    dep_data = (dep_data - 0.7) / (10.0 - 0.7)
    return dep_data[..., np.newaxis]


#--- processing function
def predict_dpu(img, dpu, input_buffer, output_buffer):
    input_buffer[0][0, ...] = img
    job_id = dpu.execute_async(input_buffer, output_buffer)
    dpu.wait(job_id)
    y_pred = output_buffer[0][0, :, :, 0]
    return y_pred


#--- metrics calculation function (optional, only used if ground-truth depth is available)
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


#--- save just the raw depth map, no decorations
def save_depth_map(dep_data_m, image_path="depth_map.png", vmin=0.70, vmax=6.00, cmap="turbo"):
    plt.imsave(image_path, dep_data_m, cmap=cmap, vmin=vmin, vmax=vmax)
    print(f"|- EXPORT IMAGE: ")
    print(f"|\t|- Image path : {image_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", type=str, default="./CNN_kv260.xmodel", help="Path to the model. Default is CNN_kv260.xmodel")
    ap.add_argument("-r", "--rgb_path", type=str, required=True, help="Path to a single RGB image")
    ap.add_argument("-g", "--depth_path", type=str, default=None, help="Optional path to the matching ground-truth depth image (for metrics)")
    ap.add_argument("-out", "--output_dir", type=str, default="./output", help="Directory where all outputs (depth map, metrics) are saved. Default is ./output")
    ap.add_argument("-o", "--output_image", type=str, default="depth_map.png", help="Filename for the predicted depth map image, saved inside output_dir. Default is depth_map.png")
    ap.add_argument("-j", "--output_json", type=str, default=None, help="Optional filename to save single-image metrics as JSON inside output_dir (requires --depth_path)")
    args = ap.parse_args()

    OUTPUT_DIR = args.output_dir

    #--- make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_out_path = os.path.join(OUTPUT_DIR, args.output_image)
    json_out_path = os.path.join(OUTPUT_DIR, args.output_json) if args.output_json else None

    print("-" * 40)
    print("single_inference.py")
    print("-" * 40)

    #--- load overlay
    overlay = DpuOverlay("dpu.bit")
    overlay.load_model(args.model_path)
    print(f"|- Model: {args.model_path}")

    #--- prepare dpu and I/O tensors
    dpu = overlay.runner

    input_tensors = dpu.get_input_tensors()
    input_tensors_shape = tuple(input_tensors[0].dims)
    output_tensors = dpu.get_output_tensors()
    output_tensors_shape = tuple(output_tensors[0].dims)

    print(f"|- DPU: ")
    print(f"|\t|- input  : {input_tensors_shape}")
    print(f"|\t|- output : {output_tensors_shape}")

    input_buffer = [np.empty(input_tensors_shape, dtype=np.float32, order="C")]
    output_buffer = [np.empty(output_tensors_shape, dtype=np.float32, order="C")]

    #--- load data
    print(f"|- Data:")
    print(f"|\t|- rgb_file   : {args.rgb_path}")
    if args.depth_path:
        print(f"|\t|- depth_file : {args.depth_path}")
    print(f"|- Output directory: {OUTPUT_DIR}")

    rgb = np.array(Image.open(args.rgb_path).convert('RGB'))[..., ::-1]
    rgb = pre_process_rgb(rgb)

    dep = None
    if args.depth_path:
        dep_raw = np.array(Image.open(args.depth_path), dtype=np.float32)
        dep = pre_process_dep(dep_raw)

    #--- run inference
    dep_pred = predict_dpu(rgb, dpu, input_buffer, output_buffer)
    dep_pred_m = dep_pred * (10.0 - 0.7) + 0.7

    #--- optional metrics if ground truth was provided
    if dep is not None:
        metrics = compute_metrics_single_image(dep, dep_pred)

        print(f"|- Metrics:")
        for k, v in metrics.items():
            print(f"|\t|- {k:<6}: {v:.4f}")


    #--- save just the raw depth map
    save_depth_map(np.squeeze(dep_pred_m), image_path=image_out_path)


if __name__ == "__main__":
    main()