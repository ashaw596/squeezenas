import abc
import enum

import torch
import torch.nn as nn


class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, c_in: int, c_out: int, stride: int, affine: bool) -> nn.Module:
        pass


class IdentityOperation(Operation):
    def __init__(self):
        super().__init__()

    def __call__(self, c_in: int, c_out: int, stride: int, affine: bool) -> nn.Module:
        if stride != 1:
            return FactorizedReduce(c_in, c_out, affine=affine)
        elif c_in == c_out:
            return Identity()
        else:
            return ReLUConvBN(c_in, c_out, kernel_size=1, stride=stride, padding=0, affine=True)


class InvertedResidualOperation(Operation):
    def __init__(self, group, kernel_size, padding, expand_ratio, dilation=1):
        super().__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.padding = padding
        self.expand_ratio = expand_ratio
        self.dilation = dilation

    def __call__(self, c_in: int, c_out: int, stride: int, affine: bool) -> nn.Module:
        assert affine is True
        return InvertedResidualBlock(c_in, c_out, kernel_size=self.kernel_size, stride=stride, padding=self.padding,
                                     expand_ratio=self.expand_ratio, group=self.group, dilation=self.dilation)


class InvertedResidualSkipOperation(Operation):
    def __init__(self):
        super().__init__()
        self.identity = IdentityOperation()

    def __call__(self, c_in: int, c_out: int, stride: int, affine: bool) -> nn.Module:
        # Assumes skip connection is run in parallel to this if c_in == c_out
        if c_in == c_out and stride == 1:
            return Zero(stride)
        else:
            return self.identity(c_in, c_out, stride, affine)


class Ops(enum.Enum):
    inverse_residual_k3_e1_g1 = InvertedResidualOperation(group=1, kernel_size=3, padding=1, expand_ratio=1)
    inverse_residual_k3_e1_g2 = InvertedResidualOperation(group=2, kernel_size=3, padding=1, expand_ratio=1)
    inverse_residual_k3_e3_g1 = InvertedResidualOperation(group=1, kernel_size=3, padding=1, expand_ratio=3)
    inverse_residual_k3_e6_g1 = InvertedResidualOperation(group=1, kernel_size=3, padding=1, expand_ratio=6)

    inverse_residual_k5_e1_g1 = InvertedResidualOperation(group=1, kernel_size=5, padding=2, expand_ratio=1)
    inverse_residual_k5_e1_g2 = InvertedResidualOperation(group=2, kernel_size=5, padding=2, expand_ratio=1)
    inverse_residual_k5_e3_g1 = InvertedResidualOperation(group=1, kernel_size=5, padding=2, expand_ratio=3)
    inverse_residual_k5_e6_g1 = InvertedResidualOperation(group=1, kernel_size=5, padding=2, expand_ratio=6)

    inverse_residual_k3_e1_g1_d2 = InvertedResidualOperation(group=1, kernel_size=3, padding=2, expand_ratio=1,
                                                             dilation=2)
    inverse_residual_k3_e1_g2_d2 = InvertedResidualOperation(group=2, kernel_size=3, padding=2, expand_ratio=1,
                                                             dilation=2)
    inverse_residual_k3_e3_g1_d2 = InvertedResidualOperation(group=1, kernel_size=3, padding=2, expand_ratio=3,
                                                             dilation=2)
    inverse_residual_k3_e6_g1_d2 = InvertedResidualOperation(group=1, kernel_size=3, padding=2, expand_ratio=6,
                                                             dilation=2)

    residual_skipish = InvertedResidualSkipOperation()

    # @DynamicClassAttribute
    # def value(self) -> Operation:
    #     return super().value

    def __repr__(self):
        return super().__str__()


class InvertedResidualBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, expand_ratio, group, dilation):
        self.expand_ratio = expand_ratio
        hidden_dim = round(c_in * expand_ratio)
        super().__init__(
            # pw
            nn.Conv2d(c_in, hidden_dim, kernel_size=1, stride=1, padding=0, groups=group, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=padding, dilation=dilation,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, c_out, kernel_size=1, stride=1, padding=0, groups=group, bias=False),
            nn.BatchNorm2d(c_out),
        )


class ReLUConvBN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=c_in, bias=False),
            nn.Conv2d(c_out, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_out, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_out, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return torch.zeros_like(x)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, c_in, c_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c_out // 2, affine=affine)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c_out // 2, affine=affine)
        )

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return out
