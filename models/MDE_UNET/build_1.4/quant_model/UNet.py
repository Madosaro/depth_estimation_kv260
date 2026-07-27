# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/Conv2d[0]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/ReLU[2]/input.4
        self.module_4 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/Conv2d[3]/input.5
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/DoubleConv[inc]/Sequential[double_conv]/ReLU[5]/152
        self.module_7 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/MaxPool2d[0]/input.7
        self.module_8 = py_nndct.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.8
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.10
        self.module_11 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.11
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down1]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/192
        self.module_14 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/MaxPool2d[0]/input.13
        self.module_15 = py_nndct.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.14
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.16
        self.module_18 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.17
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down2]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/232
        self.module_21 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/MaxPool2d[0]/input.19
        self.module_22 = py_nndct.nn.Conv2d(in_channels=96, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.20
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.22
        self.module_25 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.23
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down3]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/272
        self.module_28 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/MaxPool2d[0]/input.25
        self.module_29 = py_nndct.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[0]/input.26
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[2]/input.28
        self.module_32 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/Conv2d[3]/input.29
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Down[down4]/Sequential[maxpool_conv]/DoubleConv[1]/Sequential[double_conv]/ReLU[5]/312
        self.module_35 = py_nndct.nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up4]/ConvTranspose2d[up]/322
        self.module_36 = py_nndct.nn.Cat() #UNet::UNet/Up[up4]/input.31
        self.module_37 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.32
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.34
        self.module_40 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.35
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up4]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/359
        self.module_43 = py_nndct.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up5]/ConvTranspose2d[up]/369
        self.module_44 = py_nndct.nn.Cat() #UNet::UNet/Up[up5]/input.37
        self.module_45 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.38
        self.module_47 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.40
        self.module_48 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.41
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up5]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/406
        self.module_51 = py_nndct.nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up6]/ConvTranspose2d[up]/416
        self.module_52 = py_nndct.nn.Cat() #UNet::UNet/Up[up6]/input.43
        self.module_53 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.44
        self.module_55 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.46
        self.module_56 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.47
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up6]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/453
        self.module_59 = py_nndct.nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UNet::UNet/Up[up7]/ConvTranspose2d[up]/463
        self.module_60 = py_nndct.nn.Cat() #UNet::UNet/Up[up7]/input.49
        self.module_61 = py_nndct.nn.Conv2d(in_channels=48, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[0]/input.50
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/ReLU[2]/input.52
        self.module_64 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/Conv2d[3]/input.53
        self.module_66 = py_nndct.nn.ReLU(inplace=True) #UNet::UNet/Up[up7]/DoubleConv[conv]/Sequential[double_conv]/ReLU[5]/input
        self.module_67 = py_nndct.nn.Conv2d(in_channels=24, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #UNet::UNet/OutConv[outc]/Conv2d[conv]/510

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(self.output_module_1)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_10 = self.module_10(self.output_module_8)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_13 = self.module_13(self.output_module_11)
        self.output_module_14 = self.module_14(self.output_module_13)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_17 = self.module_17(self.output_module_15)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_20 = self.module_20(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_20)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_24 = self.module_24(self.output_module_22)
        self.output_module_25 = self.module_25(self.output_module_24)
        self.output_module_27 = self.module_27(self.output_module_25)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_29 = self.module_29(self.output_module_28)
        self.output_module_31 = self.module_31(self.output_module_29)
        self.output_module_32 = self.module_32(self.output_module_31)
        self.output_module_34 = self.module_34(self.output_module_32)
        self.output_module_35 = self.module_35(self.output_module_34)
        self.output_module_36 = self.module_36(dim=1, tensors=[self.output_module_27,self.output_module_35])
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_39 = self.module_39(self.output_module_37)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_42 = self.module_42(self.output_module_40)
        self.output_module_43 = self.module_43(self.output_module_42)
        self.output_module_44 = self.module_44(dim=1, tensors=[self.output_module_20,self.output_module_43])
        self.output_module_45 = self.module_45(self.output_module_44)
        self.output_module_47 = self.module_47(self.output_module_45)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_50 = self.module_50(self.output_module_48)
        self.output_module_51 = self.module_51(self.output_module_50)
        self.output_module_52 = self.module_52(dim=1, tensors=[self.output_module_13,self.output_module_51])
        self.output_module_53 = self.module_53(self.output_module_52)
        self.output_module_55 = self.module_55(self.output_module_53)
        self.output_module_56 = self.module_56(self.output_module_55)
        self.output_module_58 = self.module_58(self.output_module_56)
        self.output_module_59 = self.module_59(self.output_module_58)
        self.output_module_60 = self.module_60(dim=1, tensors=[self.output_module_6,self.output_module_59])
        self.output_module_61 = self.module_61(self.output_module_60)
        self.output_module_63 = self.module_63(self.output_module_61)
        self.output_module_64 = self.module_64(self.output_module_63)
        self.output_module_66 = self.module_66(self.output_module_64)
        self.output_module_67 = self.module_67(self.output_module_66)
        return self.output_module_67
