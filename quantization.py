# scripts/quantize.py

import os
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataframe import get_dataframe
from src.data.dataset import Nyudepth_png
from src.models.unet import UNet
from src.models.setup import load_for_eval
from src.quantization.calibrate import build_calibration_loader, run_calibration
from src.quantization.export import run_export


def calibrate(cfg):
    print("Starting calibration...\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\t|- Device : ", device)

    #--- load float model
    model, history, paths = load_for_eval(UNet, cfg.model_name, device,vitis_ai_version=cfg.vitis_ai_version, models_root=cfg.models)
    model.eval()

    #--- calibration dataset
    print("\t|- Dataset :")
    print("\t\t|- ", end="")
    calibration_loader = build_calibration_loader(
        cfg.dataset_vitis, cfg.quantization.calibration_samples, cfg.quantization.batch_size, cfg.quantization.num_workers
    )

    #--- calibrate
    dummy_input = torch.randn([cfg.batch_size, *cfg.input_size])
    run_calibration(model, calibration_loader, dummy_input, paths["quant_model_dir"])

    return paths["quant_model_dir"]


def export(cfg):
    print("Starting export...\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\t|- Device : ", device)


    #--- load float model
    model, history, paths = load_for_eval(UNet, cfg.model_name, device, vitis_ai_version=cfg.vitis_ai_version, models_root=cfg.models)
    model.eval()

    #--- test dataset (batch_size must be 1 for DPU)
    print("\t|- Dataset :")
    print("\t\t|- ", end="")
    test_dataframe = get_dataframe(cfg.dataset_vitis, "test")
    test_dataset = Nyudepth_png(os.path.join(cfg.dataset_vitis, "test"), test_dataframe)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #--- export
    dummy_input = torch.randn([1, *cfg.input_size])
    run_export(model, test_loader, dummy_input, paths["quant_model_dir"])


def quantize(cfg, steps):
    if "calibrate" in steps:
        calibrate(cfg)
    if "export" in steps:
        export(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/quantization.yaml")
    parser.add_argument(
        "--steps", type=str, nargs="+", default=["calibrate", "export"],
        choices=["calibrate", "export"],
        help="Which stage(s) to run. Default: both.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    quantize(cfg, steps=args.steps)