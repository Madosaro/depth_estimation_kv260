import os
import torch

import numpy as np

from pytorch_nndct.apis import torch_quantizer

from src.config import load_config
from src.data.dataframe import get_dataframe
from src.data.transforms import build_transforms
from src.data.dataset import Nyudepth_png
from src.models.setup import load_for_eval
from src.models.unet import UNet
from src.evaluation.metrics import run_evaluation, compute_metrics_from_json

def format_metric(key, val):
    if np.isnan(val):
        return "N/A"
    if "Latency" in key:
        return f"{val:.2f} ms"
    elif "FPS" in key:
        return f"{val:.2f} fps"
    elif "(m)" in key:
        return f"{val:.4f} m"
    else:
        return f"{val:.4f}"
    
def evaluate(cfg):

    print("Starting evaluation...")

    print(f'\t|- ', end="")
    test_df = get_dataframe(cfg.paths.dataset_vitis, "test")
    shape_t, color_t = build_transforms(train=False)
    test_dataset = Nyudepth_png(
        os.path.join(cfg.paths.dataset_vitis, "test"), test_df, shape_t, color_t
    )
    print(f'\t|- Dataset : ')
    print(f'\t|\t|- Test : {len(test_dataset)} pairs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\t|- Device : ", device)

    float_model,  history, paths = load_for_eval(
        UNet, cfg.model_name, device,
        vitis_ai_version=cfg.vitis_ai_version,
        models_root=cfg.paths.models
    )
    float_model.eval()

    
    rand_in = torch.randn([cfg.quantization.batch_size, 3, 224, 224])
    quantizer = torch_quantizer(
        quant_mode='test', 
        module=float_model, 
        input_args=(rand_in), 
        output_dir=paths["quant_model_dir"]
    )
    quant_model = quantizer.quant_model.to(device)
    quant_model.eval()

    float_metrics = run_evaluation(float_model, cfg.model_name, device, test_dataset, "float", paths["outputs_dir"])
    quant_metrics = run_evaluation(quant_model, cfg.model_name, device, test_dataset, "quant", paths["outputs_dir"])

    json_path = os.path.join(paths["deployment_dir"], f"{cfg.model_name}_deployement.json")
    post_metrics = {}
    post_n = 0
    if os.path.exists(json_path):
        post_metrics, post_n = compute_metrics_from_json(json_path)

    print("\n" + "=" * 105)
    print(f"{'METRIC':<15} | {'FLOAT BASELINE':<16} | {'QUANTIZED INT8':<16} | {'POST-IMPL':<16} | {'DELTA (Q-F)':<12} | {'DELTA (P-Q)':<12}")
    print("=" * 105)
    
    for metric_key in float_metrics.keys():
        f_val = float_metrics[metric_key]
        q_val = quant_metrics[metric_key]
        p_val = post_metrics.get(metric_key, float("nan"))
        
        delta_qf = q_val - f_val
        
        delta_pq = p_val - q_val if not np.isnan(p_val) and not np.isnan(q_val) else float("nan")
        
        f_str = format_metric(metric_key, f_val)
        q_str = format_metric(metric_key, q_val)
        p_str = format_metric(metric_key, p_val)
        
        is_time_metric = "FPS" in metric_key or "Latency" in metric_key
        delta_qf_str = f"{delta_qf:+.4f}" if not is_time_metric else f"{delta_qf:+.2f}"
        
        if np.isnan(delta_pq):
            delta_pq_str = "N/A"
        else:
            delta_pq_str = f"{delta_pq:+.4f}" if not is_time_metric else f"{delta_pq:+.2f}"
        
        print(f"{metric_key:<15} | {f_str:<16} | {q_str:<16} | {p_str:<16} | {delta_qf_str:<12} | {delta_pq_str:<12}")
        
    print("=" * 105)
    print(f"Post-implementation stats computed over {post_n} samples from JSON.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg)