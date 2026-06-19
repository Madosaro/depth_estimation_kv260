import torch
import torch.nn as nn
import torch.nn.functional as F

class MDELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.kernel = self._create_gaussian_kernel(7, 1.5).to(device)
        
    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        g2d = g.view(1, 1, size, 1) * g.view(1, 1, 1, size)
        return g2d

    def _native_ssim(self, y_pred, y_true, mask):
        # Calcul du SSIM standardisé (Data range = 1.0 car normalisé [0:1])
        C1, C2 = 0.01**2, 0.02**2
        channels = y_pred.shape[1]
        kernel = self.kernel.expand(channels, 1, -1, -1)
        
        mu1 = F.conv2d(y_pred, kernel, padding=3, groups=channels)
        mu2 = F.conv2d(y_true, kernel, padding=3, groups=channels)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        
        sigma1_sq = F.conv2d(y_pred * y_pred, kernel, padding=3, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, kernel, padding=3, groups=channels) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, kernel, padding=3, groups=channels) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # On applique le masque pour ne moyenner que sur les zones valides
        return (ssim_map * mask).sum() / (mask.sum() + 1e-8)

    def forward(self, y_pred, y_true, w1=1.0, w2=1.0, w3=1.0, w4=1.0):
        # 1. CRÉATION DU MASQUE (On ignore le 0.0 strict et le 1.0 strict issus du clip du dataset)
        # On garde uniquement les pixels intermédiaires bien capturés
        mask = (y_true > 0.01) & (y_true < 0.99)
        mask = mask.float()
        
        if mask.sum() == 0: # Sécurité si une image est complètement vide
            mask = torch.ones_like(y_true)

        # 2. Pertes de pixels masquées (L1 et L2)
        l1_depth = torch.sum(torch.abs(y_pred - y_true) * mask) / (mask.sum() + 1e-8)
        l2_depth = torch.sum(torch.pow(y_pred - y_true, 2) * mask) / (mask.sum() + 1e-8)

        # 3. Gradients d'images masqués (Edges)
        dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        dx_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
        
        dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        dx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        
        # Masques adaptés aux dimensions réduites des gradients (-1 pixel sur l'axe concerné)
        mask_y = mask[:, :, 1:, :]
        mask_x = mask[:, :, :, 1:]

        l_edges = (torch.sum(torch.abs(dy_pred - dy_true) * mask_y) / (mask_y.sum() + 1e-8) +
                   torch.sum(torch.abs(dx_pred - dx_true) * mask_x) / (mask_x.sum() + 1e-8))

        # 4. SSIM masqué
        ssim_val = self._native_ssim(y_pred, y_true, mask)
        l_ssim = torch.clamp((1.0 - ssim_val) * 0.5, 0.0, 1.0)

        return (
            w1 * l_ssim +
            w2 * l_edges +
            w3 * l1_depth +
            w4 * l2_depth
        )
