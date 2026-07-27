# src/quantization/calibrate.py

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pytorch_nndct.apis import torch_quantizer

from src.data.dataframe import get_dataframe
from src.data.dataset import Nyudepth_png


def build_calibration_loader(dataset_path: str, n_samples: int, batch_size: int, num_workers: int = 4) -> DataLoader:
    df = get_dataframe(dataset_path, "train").iloc[:n_samples].reset_index(drop=True)
    dataset = Nyudepth_png(dataset_path + "/train", df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_calibration(model, calibration_loader: DataLoader, dummy_input: torch.Tensor, quant_model_path: str):
    quantizer = torch_quantizer('calib', model, (dummy_input,), output_dir=quant_model_path)
    quantized_model = quantizer.quant_model

    with torch.no_grad():
        for images, _ in tqdm(calibration_loader, desc="Calibration"):
            _ = quantized_model(images)

    quantizer.export_quant_config()
    print("-" * 40)
    print(f"Calibration complete : {quant_model_path}")

    return quantized_model