import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


def conv3x3_bn(c_in, c_out, stride):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )
