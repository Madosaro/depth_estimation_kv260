# src/quantization/export.py

import torch
from torch.utils.data import DataLoader

from pytorch_nndct.apis import torch_quantizer


def run_export(model, test_loader: DataLoader, dummy_input: torch.Tensor, quant_model_path: str):
    """
    Loads the calibrated quant config, runs a single forward pass to trace it,
    and exports the xmodel for DPU deployment. batch_size of test_loader must be 1.
    """
    quantizer = torch_quantizer('test', model, (dummy_input,), output_dir=quant_model_path)
    quantized_model = quantizer.quant_model
    quantized_model.eval()

    with torch.no_grad():
        images, _ = next(iter(test_loader))
        _ = quantized_model(images)

    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model_path)
    print("-" * 40)
    print("XMODEL export complete :", quant_model_path)