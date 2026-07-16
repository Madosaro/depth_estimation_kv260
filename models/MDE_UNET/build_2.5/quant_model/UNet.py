# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/ReLU[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/Conv2d[3]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/ReLU[5]/3532
        self.module_5 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/MaxPool2d[0]/input.13
        self.module_6 = py_nndct.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.15
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.19
        self.module_8 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.21
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/3598
        self.module_10 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/MaxPool2d[0]/input.25
        self.module_11 = py_nndct.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.27
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.31
        self.module_13 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.33
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/3664
        self.module_15 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/MaxPool2d[0]/input.37
        self.module_16 = py_nndct.nn.Conv2d(in_channels=96, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.39
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.43
        self.module_18 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.45
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/3730
        self.module_20 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/MaxPool2d[0]/input.49
        self.module_21 = py_nndct.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.51
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.55
        self.module_23 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.57
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/3796
        self.module_25 = py_nndct.nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up4]/ConvTranspose2d[up]/3815
        self.module_26 = py_nndct.nn.Cat() #UNet::UNet/Up[up4]/input.61
        self.module_27 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.63
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.67
        self.module_29 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.69
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/3870
        self.module_31 = py_nndct.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up5]/ConvTranspose2d[up]/3889
        self.module_32 = py_nndct.nn.Cat() #UNet::UNet/Up[up5]/input.73
        self.module_33 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.75
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.79
        self.module_35 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.81
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/3944
        self.module_37 = py_nndct.nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up6]/ConvTranspose2d[up]/3963
        self.module_38 = py_nndct.nn.Cat() #UNet::UNet/Up[up6]/input.85
        self.module_39 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.87
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.91
        self.module_41 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.93
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/4018
        self.module_43 = py_nndct.nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up7]/ConvTranspose2d[up]/4037
        self.module_44 = py_nndct.nn.Cat() #UNet::UNet/Up[up7]/input.97
        self.module_45 = py_nndct.nn.Conv2d(in_channels=48, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.99
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.103
        self.module_47 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.105
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/input
        self.module_49 = py_nndct.nn.Conv2d(in_channels=24, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/OutConv[outc]/Conv2d[conv]/4111

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_5 = self.module_7(output_module_5)
        output_module_5 = self.module_8(output_module_5)
        output_module_5 = self.module_9(output_module_5)
        output_module_10 = self.module_10(output_module_5)
        output_module_10 = self.module_11(output_module_10)
        output_module_10 = self.module_12(output_module_10)
        output_module_10 = self.module_13(output_module_10)
        output_module_10 = self.module_14(output_module_10)
        output_module_15 = self.module_15(output_module_10)
        output_module_15 = self.module_16(output_module_15)
        output_module_15 = self.module_17(output_module_15)
        output_module_15 = self.module_18(output_module_15)
        output_module_15 = self.module_19(output_module_15)
        output_module_20 = self.module_20(output_module_15)
        output_module_20 = self.module_21(output_module_20)
        output_module_20 = self.module_22(output_module_20)
        output_module_20 = self.module_23(output_module_20)
        output_module_20 = self.module_24(output_module_20)
        output_module_20 = self.module_25(output_module_20)
        output_module_26 = self.module_26(dim=1, tensors=[output_module_15,output_module_20])
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_26 = self.module_30(output_module_26)
        output_module_26 = self.module_31(output_module_26)
        output_module_32 = self.module_32(dim=1, tensors=[output_module_10,output_module_26])
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(output_module_32)
        output_module_32 = self.module_36(output_module_32)
        output_module_32 = self.module_37(output_module_32)
        output_module_38 = self.module_38(dim=1, tensors=[output_module_5,output_module_32])
        output_module_38 = self.module_39(output_module_38)
        output_module_38 = self.module_40(output_module_38)
        output_module_38 = self.module_41(output_module_38)
        output_module_38 = self.module_42(output_module_38)
        output_module_38 = self.module_43(output_module_38)
        output_module_44 = self.module_44(dim=1, tensors=[output_module_0,output_module_38])
        output_module_44 = self.module_45(output_module_44)
        output_module_44 = self.module_46(output_module_44)
        output_module_44 = self.module_47(output_module_44)
        output_module_44 = self.module_48(output_module_44)
        output_module_44 = self.module_49(output_module_44)
        return output_module_44
