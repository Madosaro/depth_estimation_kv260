# src/quantization/paths.py

import os


def get_quantization_paths(model_name: str, vitis_ai_version: float, models_root: str = "./models/") -> dict:
    model_dir_path = os.path.join(models_root, model_name)
    build_dir_path = os.path.join(model_dir_path, f"build_{vitis_ai_version}")
    quant_model_path = os.path.join(build_dir_path, "quant_model")

    os.makedirs(build_dir_path, exist_ok=True)
    os.makedirs(quant_model_path, exist_ok=True)

    return dict(
        model_dir_path=model_dir_path,
        build_dir_path=build_dir_path,
        quant_model_path=quant_model_path,
    )