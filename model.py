# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:41:00 2023

@author: nadja
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import LION.CTtools.ct_geometry as ctgeo
from LION.models.LIONmodel import LIONmodel
from LION.utils.parameter import LIONParameter
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam






class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        # �but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_N2I(nn.Module):
    def __init__(self, n_channels, n_classes, n_features=64):
        super(UNet_N2I, self).__init__()
        self.inc = inconv(n_channels, n_features)
        self.down1 = down(n_features, 2 * n_features)
        self.down2 = down(2 * n_features, 4 * n_features)
        self.down3 = down(4 * n_features, 8 * n_features)
        self.down4 = down(8 * n_features, 8 * n_features)
        self.up1 = up(16 * n_features, 4 * n_features)
        self.up2 = up(8 * n_features, 2 * n_features)
        self.up3 = up(4 * n_features, n_features)
        self.up4 = up(2 * n_features, n_features)
        self.outc = outconv(n_features, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        Hp, Wp = ((-H % 16), (-W % 16))
        padding = (Wp // 2, Wp - Wp // 2, Hp // 2, Hp - Hp // 2)
        reflect = nn.ReflectionPad2d(padding)
        x = reflect(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        H2 = H + padding[2] + padding[3]
        W2 = W + padding[0] + padding[1]
        return x[:, :, padding[2] : H2 - padding[3], padding[0] : W2 - padding[1]]

    def clear_buffers(self):
        pass



class DummyModel(LIONmodel):
    def __init__(
        self, model_parameters: LIONParameter, geometry_parameters: ctgeo.Geometry
    ):

        super().__init__(model_parameters, geometry_parameters)  # initialize LIONmodel

        # you may want to e.g. have pytorch compatible operators
        self._make_operator()  # done!
        self.layer = torch.nn.Conv2d(
            self.model_parameters.channel_in,
            self.model_parameters.channel_out,
            3,
            padding=1,
            bias=False,
        )

    # You must define this method
    @staticmethod
    def defeult_parameters():
        param = Parameter()
        param.channel_in = 1
        param.channel_out = 1
        return param

    def forward(self, sino):
        sino_conv = self.layer(sino)
        out = self.AT(
            sino_conv  # pytorch will know how to autograd this, because of _make_operator()
        )
        return out



### Function needed when defining the UNet encoding and decoding parts
def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)


class DownConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DownConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
### Class for encoding part of the UNet. In other words, this is the part of
### the UNet which goes down with maxpooling.
class encoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels

        self.convs_and_relus1 = double_conv_and_ReLU(self.in_channels, out_channels=32)
        self.down1 = DownConvolution(in_channels=32, out_channels=32)        
      #  self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=32, out_channels=64)
        self.down2 = DownConvolution(in_channels=64, out_channels=64)        

       # self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus3 = double_conv_and_ReLU(in_channels=64, out_channels=128)

    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g):
        g_start = g
        encoding_features = []
        g = self.convs_and_relus1(g)
        encoding_features.append(g)
        g = self.down1(g)
       # g = self.maxpool1(g)
        g = self.convs_and_relus2(g)
        encoding_features.append(g)
        g = self.down2(g)
        #g = self.maxpool2(g)
        g = self.convs_and_relus3(g)

        return g, encoding_features, g_start

### Class for decoding part of the UNet. This is the part of the UNet which
### goes back up with transpose of the convolution
class decoding(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        ### Defining instance variables
        self.out_channels = out_channels

        self.transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus1 = double_conv_and_ReLU(in_channels=128, out_channels=64)
        self.transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=64, out_channels=32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)

    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g, encoding_features, g_start):
        g = self.transpose1(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus1(g)
        g = self.transpose2(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus2(g)
        g = self.final_conv(g)
        g =  g

        #g = g_start + g

        return g

### Class for the UNet model itself
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)
    ### Must have forward function. Calling encoder and deoder classes here
    ### and making the whole UNet model
    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g
    
    
### Class for the UNet model itself
class UNet_interpolation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)
    ### Must have forward function. Calling encoder and deoder classes here
    ### and making the whole UNet model
    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)
        self.relu = nn.ReLU()
     #   g = self.relu(g)
        return g