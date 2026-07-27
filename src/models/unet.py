# src/models/unet.py

import torch.nn as nn

from src.models.blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, out_activation: nn.Module = None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.params = 24

        self.inc = DoubleConv(n_channels, self.params)
        self.down1 = Down(self.params, self.params * 2)
        self.down2 = Down(self.params * 2, self.params * 4)
        self.down3 = Down(self.params * 4, self.params * 8)
        self.down4 = Down(self.params * 8, self.params * 16)

        factor = 2 if bilinear else 1

        self.up4 = Up(self.params * 16, self.params * 8 // factor, bilinear)
        self.up5 = Up(self.params * 8, self.params * 4 // factor, bilinear)
        self.up6 = Up(self.params * 4, self.params * 2 // factor, bilinear)
        self.up7 = Up(self.params * 2, self.params, bilinear)
        self.outc = OutConv(self.params, n_classes, activation=out_activation)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        logits = self.outc(x)
        return logits