import torch
import torch.nn as nn

from torchmetrics.functional.image import image_gradients
from torchmetrics.image import StructuralSimilarityIndexMeasure


class MDELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=2.0,
            reduction="elementwise_mean",
            k1=0.01,
            k2=0.02,
            kernel_size=7,
            sigma=1.5
        ).to(device)
    
    def forward(self, y_pred, y_true, w1=1.0, w2=1.0, w3=1.0, w4=1.0):
        l1_depth = torch.mean(torch.abs(y_pred - y_true))

        l2_depth = torch.mean(torch.pow(y_pred - y_true, 2))

        dy_true, dx_true = image_gradients(y_true)
        dy_pred, dx_pred = image_gradients(y_pred)
        
        l_edges = torch.mean(
            torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true)
        )

        l_ssim = torch.clamp((1.0 - self.ssim(y_pred, y_true)) * 0.5, 0.0, 1.0)

        return (
            w1 * l_ssim +
            w2 * l_edges +
            w3 * l1_depth +
            w4 * l2_depth
        )