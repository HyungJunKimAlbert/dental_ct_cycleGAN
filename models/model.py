import os
import numpy as np

import torch
import torch.nn as nn

from models.layers import *

# cycleGAN 
# https://arxiv.org/pdf/1703.10593.pdf

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="inorm"):
        super(CycleGAN, self).__init__()
    # Encoder
        self.enc1 = CBR2d(in_channels, 1*nker, kernel_size=7, stride=1, padding=3, norm=norm, relu=0.0)     # c7s1-64
        self.enc2 = CBR2d(1*nker, 2*nker, kernel_size=3, stride=2, padding=1, norm=norm, relu=0.0)          # d128
        self.enc3 = CBR2d(2*nker, 4*nker, kernel_size=3, stride=2, padding=1, norm=norm, relu=0.0)          # d256 
    # ResBlock ==> 9 R blocks in article
        res = []
        for i in range(9):
            res += [ResBlock(4*nker, 4*nker, kernel_size=3, stride=1, padding=1, norm=norm, relu=0.0)]   # R256 * 9 layers
        self.res = nn.Sequential(*res)
    # Decoder
        self.dec3 = DECBR2d(4*nker, 2*nker, kernel_size=3, stride=2, padding=1, norm=norm, relu=0.0) # u128
        self.dec2 = DECBR2d(2*nker, 1*nker, kernel_size=3, stride=2, padding=1, norm=norm, relu=0.0) # u64
        self.dec1 = CBR2d(1*nker, out_channels=1, kernel_size=7, stride=1, padding=3, relu=None, norm=None)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x
        

 
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bonrm'):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(in_channels=1*in_channels, out_channels=1*nker, kernel_size=4, stride=2, padding=1, norm=None, relu=0.2, bias=False)
        self.enc2 = CBR2d(in_channels=1*nker, out_channels=2*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(in_channels=2*nker, out_channels=4*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(in_channels=4*nker, out_channels=8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(in_channels=8*nker, out_channels=out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)
        
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        
        x = torch.sigmoid(x)

        return x




# Pix2Pix
# https://arxiv.org/pdf/1611.07004.pdf
class Pix2Pix(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Pix2Pix, self).__init__()
        # Encoder  
        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=4, padding=1,
                          norm=None, relu=0.2, stride=2)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc5 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc6 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc7 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)

        self.enc8 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1,
                          norm=norm, relu=0.2, stride=2)
        # Decoder
        self.dec1 = DECBR2d(in_channels=8*nker, out_channels=8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.drop1 = nn.Dropout2d(0.5)
        
        self.dec2 = DECBR2d(in_channels=2*8*nker, out_channels=8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.drop2 = nn.Dropout2d(0.5)

        self.dec3 = DECBR2d(in_channels=2*8*nker, out_channels=8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.drop3 = nn.Dropout2d(0.5)

        self.dec4 = DECBR2d(in_channels=2*8*nker, out_channels=8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.dec5 = DECBR2d(in_channels=2*8*nker, out_channels=4*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.dec6 = DECBR2d(in_channels=2*4*nker, out_channels=2*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.dec7 = DECBR2d(in_channels=2*2*nker, out_channels=1*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0) 
        self.dec8 = DECBR2d(in_channels=2*1*nker, out_channels=out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None) 
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        # print(f"enc1: {enc1.shape}")
        enc2 = self.enc2(enc1)
        # print(f"enc2: {enc2.shape}")
        enc3 = self.enc3(enc2)
        # print(f"enc3: {enc3.shape}")
        enc4 = self.enc4(enc3)
        # print(f"enc4: {enc4.shape}")
        enc5 = self.enc5(enc4)
        # print(f"enc5: {enc5.shape}")
        enc6 = self.enc6(enc5)
        # print(f"en67: {enc6.shape}")
        enc7 = self.enc7(enc6)
        # print(f"enc7: {enc7.shape}")
        enc8 = self.enc8(enc7)
        # print(f"enc8: {enc8.shape}")

        # Decoder
        dec1 = self.dec1(enc8)
        drop1 = self.drop1(dec1)

        cat2 = torch.cat((drop1, enc7), dim=1)
        dec2 = self.dec2(cat2)
        drop2 = self.drop2(dec2)

        cat3 = torch.cat((drop2, enc6), dim=1)
        dec3 = self.dec3(cat3)
        drop3 = self.drop3(dec3)

        cat4 = torch.cat((drop3, enc5), dim=1)
        dec4 = self.dec4(cat4)

        cat5 = torch.cat((dec4, enc4), dim=1)
        dec5 = self.dec5(cat5)

        cat6 = torch.cat((dec5, enc3), dim=1)
        dec6 = self.dec6(cat6)

        cat7 = torch.cat((dec6, enc2), dim=1)
        dec7 = self.dec7(cat7)

        cat8 = torch.cat((dec7, enc1), dim=1)
        dec8 = self.dec8(cat8)

        x = torch.tanh(dec8)
        
        return x 

        


# DC-GAN
class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN, self).__init__()

        self.dec1 = DECBR2d(in_channels=1*in_channels, out_channels=8*nker, kernel_size=4, stride=1, padding=0, norm=norm, relu=0.0, bias=False)
        self.dec2 = DECBR2d(in_channels=8*nker, out_channels=4*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec3 = DECBR2d(in_channels=4*nker, out_channels=2*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec4 = DECBR2d(in_channels=2*nker, out_channels=nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec5 = DECBR2d(in_channels=1*nker, out_channels=out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        x = torch.tanh(x)

        return x 



# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class SRResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(SRResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)
        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        # ps1 = []
        # ps1 += [nn.Conv2d(in_channels=nker, out_channels=nker, kernel_size=3, stride=1, padding=1)]
        # ps1 += [nn.ReLU()]
        # self.ps1 = nn.Sequential(*ps1)
        #
        # ps2 = []
        # ps2 += [nn.Conv2d(in_channels=nker, out_channels=nker, kernel_size=3, stride=1, padding=1)]
        # ps2 += [nn.ReLU()]
        # self.ps2 = nn.Sequential(*ps2)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = CBR2d(nker, out_channels, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=None)

    def forward(self, x):
        x = self.enc(x)
        x0 = x

        x = self.res(x)

        x = self.dec(x)
        x = x + x0

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x
