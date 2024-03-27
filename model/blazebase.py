import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, inp_chns, out_chns, activation='None'):
        super().__init__()

        if activation == 'None':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv2d(inp_chns, out_chns, 1, bias=False)
    
    def forward(self, x):
        return self.activation(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp_chns, out_chns, kernel_size, stride=1) -> None:
        super().__init__()
    
        self.conv_1 =  nn.Conv2d(inp_chns, inp_chns, kernel_size, stride=stride, groups=inp_chns, padding=kernel_size//2, bias=False)
        self.conv_2 = nn.Conv2d(inp_chns, out_chns, 1, bias=False)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))    


class SqueezeChannels(nn.Module):
    def __init__(self, inp_chns, out_chns) -> None:
        super().__init__()

        if inp_chns != out_chns:
            self.conv = nn.Conv2d(inp_chns, out_chns, 1, bias=False)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        return self.conv(x)


class DecoderConv(nn.Module):
    def __init__(self, inp_chns, out_chns, block_num, sampler=None) -> None:
        super().__init__()

        self.sampler = sampler

        if self.sampler == 'down':
            self.conv_b = DepthwiseSeparableConv(inp_chns, out_chns, 3, stride=2)
        else:
            self.conv_b = DepthwiseSeparableConv(inp_chns, out_chns, 3)
        
        self.conv = nn.ModuleList()
        for i in range(block_num):
            self.conv.append(DepthwiseSeparableConv(out_chns, out_chns, kernel_size=3))

    def forward(self, x):
        x = F.relu(self.conv_b(x))

        for i in range(len(self.conv)):
            x = F.relu(x + self.conv[i](x))

        if self.sampler == 'up':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x


class ChannelPadding(nn.Module):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def forward(self, x):
        pad_shape = (0, 0, 0, 0, 0, self.channels - x.size(1))
        out = nn.functional.pad(x, pad_shape, 'constant', 0)

        return out

class BlazeBlock(nn.Module):
    def __init__(self, inp_channel, out_channel, block_num=3):
        super(BlazeBlock, self).__init__()

        self.downsample_a = DepthwiseSeparableConv(inp_channel, out_channel, 3, stride=2)   

        if inp_channel != out_channel:
            self.downsample_b = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ChannelPadding(channels=out_channel)
            )
        else:
            self.downsample_b = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = nn.ModuleList()
        for i in range(block_num):
            self.conv.append(DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3))

    def forward(self, x):
        da = self.downsample_a(x)
        db = self.downsample_b(x)
        
        x = F.relu(da + db)

        for i in range(len(self.conv)):
            x = F.relu(x + self.conv[i](x))
        return x
