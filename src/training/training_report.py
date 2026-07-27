# src/training/training_report.py

import os

import matplotlib.pyplot as plt

from src.models.setup import load_for_eval


COMPONENT_COLORS = {
    'MAE': 'orange',
    'MSE': 'red',
    'Edge': 'magenta',
    'SSIM': 'teal',
}

COMPONENT_KEYS = {
    'MAE': ('MAE_train', 'MAE_val'),
    'MSE': ('MSE_train', 'MSE_val'),
    'Edge': ('Edge_train', 'Edge_val'),
    'SSIM': ('SSIM_train', 'SSIM_val'),
}


def generate_training_report(history, output_dir):

    print("Starting Training report generation ...")

    num_epochs = len(history['loss_train'])
    epochs = range(1, num_epochs + 1)

    training_time = sum(history['epoch_time']) / 60.0

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4)

    fig.suptitle(
        f"Training Report — {num_epochs} epochs — {training_time:.1f} min "
        f"({training_time / num_epochs:.2f} min/epoch)",
        fontsize=16,
        fontweight="bold",
    )

    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_loss.plot(epochs, history['loss_train'], label='Train Loss', linestyle=':', color='steelblue')
    ax_loss.plot(epochs, history['loss_val'], label='Val Loss', linestyle='-', color='steelblue')
    ax_loss.set_title('Global Loss over Epochs')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.5)

    ax_lr = fig.add_subplot(gs[0, 2:4])
    ax_lr.plot(epochs, history['lr'], label='Learning Rate', color='purple', linestyle='--')
    ax_lr.set_title('Learning Rate over Epochs')
    ax_lr.set_xlabel('Epochs')
    ax_lr.set_ylabel('LR')
    ax_lr.legend()
    ax_lr.grid(True, alpha=0.5)

    for i, (name, (train_key, val_key)) in enumerate(COMPONENT_KEYS.items()):
        ax = fig.add_subplot(gs[1, i])
        color = COMPONENT_COLORS[name]
        has_data = False

        if len(history[train_key]) > 0:
            ax.plot(epochs, history[train_key], label='Train', linestyle=':', color=color)
            has_data = True
        if len(history[val_key]) > 0:
            ax.plot(epochs, history[val_key], label='Val', linestyle='-', color=color)
            has_data = True

        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data\navailable',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_title(f'{name} Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Value')
        ax.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_save_path = os.path.join(output_dir, "training_report.png")
    plt.savefig(output_save_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.show()