import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, dilation=1, groups=1, stride=1, transpose:bool=False):
        super().__init__()
        if transpose is False:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding,
                                  bias=False, dilation=dilation, groups=groups, stride=stride)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, padding=padding,
                                           bias=False, dilation=dilation, groups=groups, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
