import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, mid_channels:int=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )  
    
    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  #x2 skip connection
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.params = 24

        self.inc = (DoubleConv(n_channels, self.params))
        self.down1 = (Down(self.params, self.params*2))
        self.down2 = (Down(self.params*2, self.params*4))
        self.down3 = (Down(self.params*4, self.params*8))
        self.down4 = (Down(self.params*8, self.params*16))

        factor = 2 if bilinear else 1

        self.up4 = Up(self.params * 16, self.params * 8 // factor, bilinear) 
        self.up5 = Up(self.params * 8, self.params * 4 // factor, bilinear)        
        self.up6 = Up(self.params * 4, self.params * 2 // factor, bilinear)
        self.up7 = Up(self.params * 2, self.params, bilinear)
        self.outc = (OutConv(self.params, n_classes))


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
    

class OutConv_sigmoid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
    

class UNet_sigmoid(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.params = 24

        self.inc = (DoubleConv(n_channels, self.params))
        self.down1 = (Down(self.params, self.params*2))
        self.down2 = (Down(self.params*2, self.params*4))
        self.down3 = (Down(self.params*4, self.params*8))
        self.down4 = (Down(self.params*8, self.params*16))

        factor = 2 if bilinear else 1

        self.up4 = Up(self.params * 16, self.params * 8 // factor, bilinear) 
        self.up5 = Up(self.params * 8, self.params * 4 // factor, bilinear)        
        self.up6 = Up(self.params * 4, self.params * 2 // factor, bilinear)
        self.up7 = Up(self.params * 2, self.params, bilinear)
        self.outc = (OutConv_sigmoid(self.params, n_classes))


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