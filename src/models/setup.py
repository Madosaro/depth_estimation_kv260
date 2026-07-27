# src/model/setup.py

import os
import torch

from src.training.history import History
from src.training.checkpoint import (
    resolve_history_path,
    resolve_checkpoint_path,
    load_checkpoint_file,
    apply_training_state,
    apply_model_state,
)

def build_model(model_class, device):
    print(f'\t|- Model :')
    return model_class().to(device)


def build_optimizer(model, lr=0.0001, optimizer_cls=torch.optim.AdamW):
    return optimizer_cls(model.parameters(), lr=lr)


def build_scheduler(optimizer, scheduler_kwargs=None):
    scheduler_kwargs = scheduler_kwargs or dict(mode='min', patience=3, factor=0.5)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

def get_model_paths(model_name, vitis_ai_version: float, models_root="./models/"):
    model_dir_path = os.path.join(models_root, model_name)
    model_path = os.path.join(model_dir_path, model_name + ".pth")
    history_path = os.path.join(model_dir_path, model_name + "_history.json")

    checkpoints_dir_path = os.path.join(model_dir_path, "checkpoints")
    
    outputs_dir_path = os.path.join(model_dir_path, "outputs")
    deployment_dir_path = os.path.join(outputs_dir_path, f"deployment_{vitis_ai_version}")
    
    build_dir_path = os.path.join(model_dir_path, f"build_{vitis_ai_version}")
    quant_model_dir_path = os.path.join(build_dir_path, "quant_model")

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(checkpoints_dir_path, exist_ok=True)

    os.makedirs(outputs_dir_path, exist_ok=True)
    os.makedirs(deployment_dir_path, exist_ok=True)

    os.makedirs(build_dir_path, exist_ok=True)
    os.makedirs(quant_model_dir_path, exist_ok=True)

    return dict(
        model_dir=model_dir_path,
        checkpoints_dir=checkpoints_dir_path,
        outputs_dir=outputs_dir_path,
        history=history_path,
        model=model_path,
        build_dir=build_dir_path,
        quant_model_dir=quant_model_dir_path,
        deployment_dir=deployment_dir_path
    )


def load_for_training(
    model_class,
    model_name: str,
    device: str,
    resume_epoch: int,
    vitis_ai_version,
    models_root: str = "./models/",
    lr: float = 0.0001,
):
    paths = get_model_paths(model_name, vitis_ai_version, models_root)

    model = build_model(model_class, device)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = build_scheduler(optimizer)

    resume_from = None
    if resume_epoch is not None:
        resume_from = resolve_checkpoint_path(model_name, paths["checkpoints_dir"], resume_epoch)

    load_path = resume_from or paths["model"]
    checkpoint = load_checkpoint_file(load_path, device)
    start_epoch = apply_training_state(model, optimizer, scheduler, checkpoint)

    history_path = resolve_history_path(resume_from, paths["history"])
    history = History.load(history_path)
    if resume_from is not None:
        history.truncate(start_epoch)

    return model, optimizer, scheduler, start_epoch, history, paths


def load_for_eval(model_class, model_name, device, vitis_ai_version, models_root="./models/"):
    model = build_model(model_class, device)
    paths = get_model_paths(model_name, vitis_ai_version, models_root)

    checkpoint = load_checkpoint_file(paths["model"], device)
    apply_model_state(model, checkpoint)
    model.eval()

    history = History.load(paths["history"])

    return model, history, paths



def load_quantized_for_eval(model_class, model_name, device, models_root="./models/"):
    model = build_model(model_class, device)
    paths = get_model_paths(model_name, models_root)

    quant_path = os.path.join(paths["quan"], f"{model_name}_quantized.pth")
    load_path = quant_path if os.path.exists(quant_path) else paths["model"]

    # 3. Charger le fichier de checkpoint
    checkpoint = load_checkpoint_file(load_path, device)

    # 4. Préparer / Convertir le modèle à la quantification
    # Si tu as une méthode pour fusionner les couches (ex: Conv + BN + ReLU)
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
        
    # NOTE: Si tu utilises la boîte à outils PyTorch standard (FX Graph ou Eager), 
    # c'est ici qu'on applique la conversion des modules, par exemple :
    # model = torch.quantization.convert(model, inplace=True)

    # 5. Appliquer les poids quantifiés et basculer en évaluation
    apply_model_state(model, checkpoint)
    model.eval()

    # 6. Charger l'historique (avec une sécurité si le JSON n'a pas été généré pour la version quantifiée)
    try:
        history = History.load(paths["history"])
    except Exception:
        history = None

    return model, history, paths