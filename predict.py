import os
import copy
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_nndct.apis import torch_quantizer

from src.config import load_config
from src.data.dataframe import get_dataframe
from src.data.transforms import build_transforms
from src.data.dataset import Nyudepth_png
from src.models.setup import load_for_eval
from src.models.unet import UNet

DEPTH_MIN_M = 0.7
DEPTH_MAX_M = 10.0
VMIN_SHARED, VMAX_SHARED = 0.7, 6.0


def to_meters(depth_01):
    return depth_01 * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M


def load_deployment_depth_m(deployment_dir, rgb_filename):
    base_name = os.path.basename(rgb_filename)
    stem = os.path.splitext(base_name)[0]

    candidates = [
        os.path.join(deployment_dir, f"pred_{stem}.png"),
        os.path.join(deployment_dir, base_name),
    ]

    for path in candidates:
        if os.path.exists(path):
            depth_mm = np.array(Image.open(path), dtype=np.float32)
            return depth_mm / 1000.0

    return None


def make_figure(idx, filename, rgb_img, gt_depth_m, float_depth_m, quant_depth_m,
                 deploy_depth_m, error_float_m, error_quant_m, error_deploy_m,
                 valid_float, valid_quant, valid_deploy, out_path):

    cmap = copy.copy(plt.get_cmap("turbo"))
    cmap.set_bad(color="black")

    mosaic = [
        ["rgb", "gnd", "float", "quant", "deploy"],
        [".", ".", "err_float", "err_quant", "err_deploy"],
        ["hist", "hist", "hist", "hist", "hist"],
    ]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(22, 12))

    #--- RGB
    axs["rgb"].set_title(f"Input RGB (Idx: {idx} - {filename})")
    axs["rgb"].imshow(rgb_img)
    axs["rgb"].axis("off")

    # --- Échelle partagée ---
    ticks_m = np.linspace(VMIN_SHARED, VMAX_SHARED, 6)
    tick_labels = [f"{t:.2f}m" for t in ticks_m]

    def plot_depth(ax_key, data, title):
        axs[ax_key].set_title(title)
        im = axs[ax_key].imshow(data, cmap=cmap, vmin=VMIN_SHARED, vmax=VMAX_SHARED)
        axs[ax_key].axis("off")
        cbar = fig.colorbar(im, ax=axs[ax_key], orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_ticks(ticks_m)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("Distance (m)", rotation=270, labelpad=15)

    #--- Depth
    plot_depth("gnd", gt_depth_m, "Ground Truth Depth")

    #--- Float
    plot_depth("float", float_depth_m, "Float Model Prediction")

    #--- Quant
    plot_depth("quant", quant_depth_m, "Quantized Model Prediction")

    #--- Deployment
    if deploy_depth_m is not None:
        plot_depth("deploy", deploy_depth_m, "Deployment Model Prediction")
    else:
        axs["deploy"].set_title("Deployment Model Prediction (N/A)")
        axs["deploy"].axis("off")

    #--- Absolute errors
    def plot_error(ax_key, data, title):
        axs[ax_key].set_title(title)
        im = axs[ax_key].imshow(data, cmap="jet", vmin=0, vmax=VMAX_SHARED)
        axs[ax_key].axis("off")
        fig.colorbar(im, ax=axs[ax_key], orientation="vertical", fraction=0.046, pad=0.04)

    plot_error("err_float", error_float_m, "Absolute Error (Float)")
    plot_error("err_quant", error_quant_m, "Absolute Error (Quantized)")

    if error_deploy_m is not None:
        plot_error("err_deploy", error_deploy_m, "Absolute Error (Deployment)")
    else:
        axs["err_deploy"].set_title("Absolute Error (Deployment) (N/A)")
        axs["err_deploy"].axis("off")

    #--- Histogram
    mae_float = np.mean(valid_float)
    mae_quant = np.mean(valid_quant)

    title = (f"Error Distribution (Idx: {idx}) - "
             f"Float MAE: {mae_float:.3f}m | Quantized MAE: {mae_quant:.3f}m")

    axs["hist"].hist(valid_float, bins=120, color="royalblue", edgecolor="black", alpha=0.5, label="Float")
    axs["hist"].hist(valid_quant, bins=120, color="seagreen", edgecolor="black", alpha=0.5, label="Quantized")
    axs["hist"].axvline(mae_float, color="blue", linestyle="dashed", linewidth=2, label=f"Mean Float : {mae_float:.3f}m")
    axs["hist"].axvline(mae_quant, color="green", linestyle="dashed", linewidth=2, label=f"Mean Quantized : {mae_quant:.3f}m")

    if valid_deploy is not None and len(valid_deploy) > 0:
        mae_deploy = np.mean(valid_deploy)
        title += f" | Deployment MAE: {mae_deploy:.3f}m"
        axs["hist"].hist(valid_deploy, bins=120, color="darkorange", edgecolor="black", alpha=0.4, label="Deployment")
        axs["hist"].axvline(mae_deploy, color="darkorange", linestyle="dashed", linewidth=2,
                             label=f"Mean Deployment : {mae_deploy:.3f}m")
    else:
        title += " | Deployment: N/A"

    axs["hist"].set_title(title)
    axs["hist"].set_xlabel("Absolute error (m)", fontsize=11)
    axs["hist"].set_ylabel("Number of pixels", fontsize=11)
    axs["hist"].set_xlim(0, DEPTH_MAX_M - DEPTH_MIN_M)
    axs["hist"].grid(axis="y", alpha=0.3)
    axs["hist"].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(cfg_path, max_images, start_index, deployment_dir_override):
    cfg = load_config(cfg_path)

    print("Starting per-image plot generation...")

    test_df = get_dataframe(cfg.paths.dataset_vitis, "test")
    shape_t, color_t = build_transforms(train=False)
    test_dataset = Nyudepth_png(
        os.path.join(cfg.paths.dataset_vitis, "test"), test_df, shape_t, color_t
    )
    print(f"\t|- Dataset : Test : {len(test_dataset)} pairs")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\t|- Device : ", device)

    float_model, history, paths = load_for_eval(
        UNet, cfg.model_name, device,
        vitis_ai_version=cfg.vitis_ai_version,
        models_root=cfg.paths.models
    )
    float_model.eval()

    rand_in = torch.randn([cfg.quantization.batch_size, 3, 224, 224])
    quantizer = torch_quantizer(
        quant_mode="test",
        module=float_model,
        input_args=(rand_in),
        output_dir=paths["quant_model_dir"]
    )
    quant_model = quantizer.quant_model.to(device)
    quant_model.eval()

    deployment_dir = deployment_dir_override or os.path.join(paths["deployment_dir"], "deployment_data")
    print(f"\t|- Deployment data dir : {deployment_dir}")
    if not os.path.isdir(deployment_dir):
        print(f"[warn] deployment data directory not found, deployment panels will be skipped: {deployment_dir}")

    out_dir = os.path.join(paths["deployment_dir"], "visual_results")
    os.makedirs(out_dir, exist_ok=True)

    n_total = len(test_dataset)
    end_index = n_total if max_images is None else min(n_total, start_index + max_images)

    for idx in range(start_index, end_index):
        images, depth_gt = test_dataset[idx]
        images_b = images.unsqueeze(0).to(device)

        with torch.no_grad():
            float_pred = float_model(images_b)
            quant_pred = quant_model(images_b)

        rgb_img = images.cpu().permute(1, 2, 0).numpy()
        rmin, rmax = rgb_img.min(), rgb_img.max()
        if rmax > rmin:
            rgb_img = (rgb_img - rmin) / (rmax - rmin)

        gt_np = depth_gt.cpu().squeeze().numpy()
        float_np = float_pred[0].cpu().squeeze().numpy()
        quant_np = quant_pred[0].cpu().squeeze().numpy()

        gt_m = to_meters(gt_np)
        float_m = np.clip(to_meters(float_np), DEPTH_MIN_M, DEPTH_MAX_M)
        quant_m = np.clip(to_meters(quant_np), DEPTH_MIN_M, DEPTH_MAX_M)

        mask = (gt_np > 0.01) & (gt_np < 0.99)
        error_float_m = np.abs(float_m - gt_m)
        error_quant_m = np.abs(quant_m - gt_m)
        valid_float = error_float_m[mask]
        valid_quant = error_quant_m[mask]

        filename = test_df.iloc[idx]["rgb"]
        base_name = os.path.basename(filename)

        deploy_m = load_deployment_depth_m(deployment_dir, base_name)
        error_deploy_m = None
        valid_deploy = None
        if deploy_m is not None:
            if deploy_m.shape == gt_m.shape:
                deploy_m = np.clip(deploy_m, DEPTH_MIN_M, DEPTH_MAX_M)
                error_deploy_m = np.abs(deploy_m - gt_m)
                valid_deploy = error_deploy_m[mask]
            else:
                print(f"[warn] shape mismatch for {base_name}: "
                      f"deployment {deploy_m.shape} vs ground truth {gt_m.shape}, skipping deployment panels")
                deploy_m = None

        out_path = os.path.join(
            out_dir, f"comparison_idx_{idx}_{os.path.splitext(base_name)[0]}.png"
        )
        make_figure(idx, base_name, rgb_img, gt_m, float_m, quant_m, deploy_m,
                    error_float_m, error_quant_m, error_deploy_m,
                    valid_float, valid_quant, valid_deploy, out_path)
        print(f"[{idx + 1}/{end_index}] saved -> {out_path}")

    print(f"Done. {end_index - start_index} figures written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-images", type=int, default=None,
                         help="Limit the number of images processed (default: all test images)")
    parser.add_argument("--start-index", type=int, default=0,
                         help="Index to start from in the test dataset")
    parser.add_argument("--deployment-dir", type=str, default=None,
                         help="Override path to the deployment_data folder "
                              "(default: <model outputs dir>/deployment_data)")
    args = parser.parse_args()

    main(args.config, args.max_images, args.start_index, args.deployment_dir)