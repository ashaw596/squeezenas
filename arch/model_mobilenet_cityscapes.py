# noinspection PyPep8Naming
import copy
from dataclasses import dataclass
import dataclasses
from typing import Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from arch.common import Conv_BN_ReLU
from arch.model_mobilenet import DSMobileNetBlockMetaHyperparameters, DSMobileNet, \
    DSMobileNetMetaHyperparameters
from arch.operations import Ops

@dataclass(frozen=True)
class DSMobileNetMetaCityscapesHyperparameters:
    init_channels: int
    blocks: Sequence[DSMobileNetBlockMetaHyperparameters]
    num_classes: int
    skip_output_block_index: int
    mid_channels: int
    last_channels: Optional[int] = None

    def to_ds_mobile_net_hyperparamters(self, last_channels, num_classes, last_pooled_channels=None):
        if self.last_channels is not None:
            assert last_channels is None
        if self.last_channels is not None:
            last_channels = self.last_channels
        return DSMobileNetMetaHyperparameters(init_channels=self.init_channels,
                                              blocks=copy.deepcopy(tuple(self.blocks)),
                                              last_channels=last_channels,
                                              num_classes=num_classes, last_pooled_channels=last_pooled_channels)

@dataclass(frozen=True)
class MetaNetworkParams:
    hyperparameters: DSMobileNetMetaCityscapesHyperparameters
    genotype: Sequence[Ops]
    weight_path: Optional[str] = None


class ASPP(nn.Module):
    def __init__(self, in_ch:int, mid_ch:int, out_ch:int, rates=(6,12,18), groups=(1,1,1)):
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
        avg_pooled = F.avg_pool2d(x, (h, w), stride=(1,1), padding=0, ceil_mode=False, count_include_pad=False)
        img_pool = F.interpolate(avg_pooled, size=(h, w), mode='nearest')

        tmp6 = torch.cat([tmp1, tmp2, tmp3, tmp4, img_pool], dim=1)

        return self._1x1_2_conv(tmp6)


class ASPP_Lite(nn.Module):
    def __init__(self, os16_channels, os8_channels, mid_channels, num_classes:int):
        super().__init__()

        self._1x1_TL = Conv_BN_ReLU(os16_channels, mid_channels, kernel=1)
        self._1x1_BL = nn.Conv2d(os16_channels, mid_channels, kernel_size=1) # TODO: bias=False?
        self._1x1_TR = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self._1x1_BR = nn.Conv2d(os8_channels, num_classes, kernel_size=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=49, stride=[16,20], count_include_pad=False)

    def forward(self, OS16, OS8):
        assert OS16.shape[-1] * 2 == OS8.shape[-1], (OS8.shape, OS16.shape)

        t1 = self._1x1_TL(OS16)
        B,C,H,W = t1.shape
        t2 = self.avgpool(OS16)
        t2 = self._1x1_BL(t2)
        t2 = torch.sigmoid(t2)
        t2 = F.interpolate(t2, size=(H, W), mode='bilinear', align_corners=False)
        t3 = t1 * t2
        t3 = F.interpolate(t3, scale_factor=2, mode='bilinear', align_corners=False)
        t3 = self._1x1_TR(t3)
        t4 = self._1x1_BR(OS8)
        return t3 + t4

class DSMobileNetMetaCityscapes(nn.Module):
    @staticmethod
    def from_meta_params(meta_params: MetaNetworkParams):
        model = DSMobileNetMetaCityscapes(hyperparams=meta_params.hyperparameters, genotype=meta_params.genotype)

        if meta_params.weight_path is not None:
            dic = torch.load(meta_params.weight_path)
            state_dict = dic['state_dict']
            new_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[len('module.'):]] = v
                else:
                    new_state_dict[k] = v

            model.cuda()
            new_state_dict = {k: v for k, v in new_state_dict.items()
                              if
                              not (k.startswith('last_conv.') or k.startswith('classifier.') or k.startswith(
                                  'criterion.'))}

            model.encoder.load_state_dict(new_state_dict)
        return model

    def __init__(self, hyperparams: DSMobileNetMetaCityscapesHyperparameters, genotype: Sequence[Ops], lr_aspp=True):
        super().__init__()
        self.hyperparams = hyperparams
        self.genotype = genotype
        self.lr_aspp = lr_aspp

        self.encoder = DSMobileNet(hyperparams=hyperparams.to_ds_mobile_net_hyperparamters(last_channels=None, num_classes=None),
                                   genotype=genotype)

        self.criterion = CrossEntropyLoss(ignore_index=255)

        mid_ch = hyperparams.mid_channels
        # last_channels = 320
        last_channels = hyperparams.blocks[-1].num_channels
        # assert hyperparams.blocks[-1].num_channels == last_channels
        # assert hyperparams.blocks[hyperparams.skip_output_block_index].num_channels == mid_ch

        count = 0
        for block in hyperparams.blocks:
            count += block.num_repeat
            if count > self.hyperparams.skip_output_block_index:
                low_level_channels = block.num_channels
                break


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

            # 24 = mobilenet mid feat channel
            # self.logits1 = Conv_BN_ReLU(mid_ch+24, mid_ch+24, 1)
            self.logits2 = Conv_BN_ReLU(mid_ch + low_level_channels, mid_ch + low_level_channels, 3,
                                        groups=mid_ch + low_level_channels, padding=1)
            self.logits3 = nn.Conv2d(low_level_channels + mid_ch, hyperparams.num_classes, kernel_size=1)
        # self.decoder = ASPP(in_ch=last_channels, mid_ch=mid_ch, out_ch=mid_ch, groups=(mid_ch,) * 3)

        # self.lowlevel1x1 = nn.Conv2d(low_level_channels, low_level_channels, 1)

        # 24 = mobilenet mid feat channel
        # self.logits1 = Conv_BN_ReLU(mid_ch+24, mid_ch+24, 1)
        # self.logits2 = Conv_BN_ReLU(mid_ch + low_level_channels, mid_ch + low_level_channels, 3,
        #                             groups=mid_ch + low_level_channels, padding=1)
        # self.logits3 = nn.Conv2d(low_level_channels + mid_ch, hyperparams.num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.export = False

        # init the last layer to be slightly more random
        # nn.init.normal_(self.logits3.weight, mean=0, std=0.1)
        # nn.init.constant_(self.logits3.bias, 0)

    def forward(self, normalized_rgb, gt=None, rgb=None, fname=None):
        encoder_outputs = self.encoder(normalized_rgb)
        residuals_outputs = encoder_outputs['residuals_outputs']
        cur_feat = encoder_outputs['output']

        b, c, h, w = cur_feat.shape
        assert w >= h, cur_feat.shape

        low_level_feat = residuals_outputs[self.hyperparams.skip_output_block_index]

        # ft1, ft2, ft3, ft4 = self.encoder(x)
        # logits = self.decoder(cur_feat)
        # logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        # low_level_feat = self.lowlevel1x1(low_level_feat)
        # logits = torch.cat((logits, low_level_feat), dim=1)
        # # x = self.logits1(x)
        # logits = self.logits2(logits)
        # logits = self.logits3(logits)
        if self.lr_aspp:
            logits = self.decoder(cur_feat, low_level_feat)
            logits = F.interpolate(logits, scale_factor=8, mode='bilinear', align_corners=True)

        else:
            # ft1, ft2, ft3, ft4 = self.encoder(x)
            logits = self.decoder(cur_feat)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
            # print(low_level_feat.size())
            low_level_feat = self.lowlevel1x1(low_level_feat)
            logits = torch.cat((logits, low_level_feat), dim=1)
            # x = self.logits1(x)
            logits = self.logits2(logits)
            logits = self.logits3(logits)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)

        if gt is None:
            return {'preds': logits}

        logits = logits.float()
        loss = self.criterion(logits, gt)
        return {'loss': loss, 'preds': logits}
