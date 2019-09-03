# noinspection PyPep8Naming
import copy
from dataclasses import dataclass
from typing import Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from arch.model import InvertResidualNetBlockMetaHyperparameters, SqueezeNASNet, \
    InverseResidualMetaNetHyperparameters
from arch.operations import Ops


@dataclass(frozen=True)
class SqueezeNASNetCityscapesHyperparameters:
    init_channels: int
    blocks: Sequence[InvertResidualNetBlockMetaHyperparameters]
    num_classes: int
    skip_output_block_index: int
    mid_channels: int
    last_channels: Optional[int] = None

    def to_ds_mobile_net_hyperparameters(self, last_channels, num_classes, last_pooled_channels=None):
        if self.last_channels is not None:
            assert last_channels is None
        if self.last_channels is not None:
            last_channels = self.last_channels
        return InverseResidualMetaNetHyperparameters(init_channels=self.init_channels,
                                                     blocks=copy.deepcopy(tuple(self.blocks)),
                                                     last_channels=last_channels,
                                                     num_classes=num_classes, last_pooled_channels=last_pooled_channels)


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, dilation=1, groups=1, stride=1,
                 transpose: bool = False):
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


class ASPP(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, rates=(6, 12, 18), groups=(1, 1, 1)):
        super().__init__()
        self._1x1_1_conv = Conv_BN_ReLU(in_ch, mid_ch, kernel=1)
        self._3x3_1_conv = Conv_BN_ReLU(in_ch, mid_ch, kernel=3, padding=rates[0], dilation=rates[0], groups=groups[0])
        self._3x3_2_conv = Conv_BN_ReLU(in_ch, mid_ch, kernel=3, padding=rates[1], dilation=rates[1], groups=groups[1])
        self._3x3_3_conv = Conv_BN_ReLU(in_ch, mid_ch, kernel=3, padding=rates[2], dilation=rates[2], groups=groups[2])
        self._1x1_2_conv = Conv_BN_ReLU(mid_ch * 4 + in_ch, out_ch, kernel=1)

    def forward(self, x):
        b, c, h, w = x.shape
        tmp1 = self._1x1_1_conv(x)
        tmp2 = self._3x3_1_conv(x)
        tmp3 = self._3x3_2_conv(x)
        tmp4 = self._3x3_3_conv(x)
        avg_pooled = F.avg_pool2d(x, (h, w), stride=(1, 1), padding=0, ceil_mode=False, count_include_pad=False)
        img_pool = F.interpolate(avg_pooled, size=(h, w), mode='nearest')
        tmp6 = torch.cat([tmp1, tmp2, tmp3, tmp4, img_pool], dim=1)
        return self._1x1_2_conv(tmp6)


class ASPP_Lite(nn.Module):
    def __init__(self, os16_channels, os8_channels, mid_channels, num_classes: int):
        super().__init__()
        self._1x1_TL = Conv_BN_ReLU(os16_channels, mid_channels, kernel=1)
        self._1x1_BL = nn.Conv2d(os16_channels, mid_channels, kernel_size=1)  # TODO: bias=False?
        self._1x1_TR = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self._1x1_BR = nn.Conv2d(os8_channels, num_classes, kernel_size=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=49, stride=[16, 20], count_include_pad=False)

    def forward(self, os16, os8):
        assert os16.shape[-1] * 2 == os8.shape[-1], (os8.shape, os16.shape)
        t1 = self._1x1_TL(os16)
        B, C, H, W = t1.shape
        t2 = self.avgpool(os16)
        t2 = self._1x1_BL(t2)
        t2 = torch.sigmoid(t2)
        t2 = F.interpolate(t2, size=(H, W), mode='bilinear', align_corners=False)
        t3 = t1 * t2
        t3 = F.interpolate(t3, scale_factor=2, mode='bilinear', align_corners=False)
        t3 = self._1x1_TR(t3)
        t4 = self._1x1_BR(os8)
        return t3 + t4


class SqueezeNASNetCityscapes(nn.Module):
    def __init__(self, hyperparams: SqueezeNASNetCityscapesHyperparameters, genotype: Sequence[Ops], lr_aspp=True):
        super().__init__()
        self.hyperparams = hyperparams
        self.genotype = genotype
        self.lr_aspp = lr_aspp

        self.encoder = SqueezeNASNet(
            hyperparams=hyperparams.to_ds_mobile_net_hyperparameters(last_channels=None, num_classes=None),
            genotype=genotype)

        self.criterion = CrossEntropyLoss(ignore_index=255)

        mid_ch = hyperparams.mid_channels

        low_level_channels = None
        count = 0
        for block in hyperparams.blocks:
            count += block.num_repeat
            if count > self.hyperparams.skip_output_block_index:
                low_level_channels = block.num_channels
                break

        assert low_level_channels is not None

        if hyperparams.last_channels:
            last_channels = hyperparams.last_channels
        else:
            last_channels = hyperparams.blocks[-1].num_channels

        if self.lr_aspp:
            self.decoder = ASPP_Lite(os16_channels=last_channels, os8_channels=low_level_channels,
                                     mid_channels=mid_ch, num_classes=hyperparams.num_classes)
        else:
            self.decoder = ASPP(in_ch=last_channels, mid_ch=mid_ch, out_ch=mid_ch, groups=(mid_ch,) * 3)
            self.lowlevel1x1 = nn.Conv2d(low_level_channels, low_level_channels, 1)
            self.logits2 = Conv_BN_ReLU(mid_ch + low_level_channels, mid_ch + low_level_channels, 3,
                                        groups=mid_ch + low_level_channels, padding=1)
            self.logits3 = nn.Conv2d(low_level_channels + mid_ch, hyperparams.num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, normalized_rgb, gt=None, rgb=None, fname=None):
        encoder_outputs = self.encoder(normalized_rgb)
        residuals_outputs = encoder_outputs['residuals_outputs']
        cur_feat = encoder_outputs['output']

        b, c, h, w = cur_feat.shape
        assert w >= h, cur_feat.shape

        low_level_feat = residuals_outputs[self.hyperparams.skip_output_block_index]

        if self.lr_aspp:
            logits = self.decoder(cur_feat, low_level_feat)
            logits = F.interpolate(logits, scale_factor=8, mode='bilinear', align_corners=True)

        else:
            logits = self.decoder(cur_feat)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
            low_level_feat = self.lowlevel1x1(low_level_feat)
            logits = torch.cat((logits, low_level_feat), dim=1)
            logits = self.logits2(logits)
            logits = self.logits3(logits)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)

        if gt is None:
            return {'preds': logits}

        logits = logits.float()
        loss = self.criterion(logits, gt)
        return {'loss': loss, 'preds': logits}
