import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from arch.model_cityscapes import SqueezeNASNetCityscapesHyperparameters, ASPP_Lite, ASPP, Conv_BN_ReLU
from search.arch_search import SuperNetwork
from search.model_search import SuperNetworkSqueezeNASNet


class SuperNetworkSqueezeNASNetCityscapes(SuperNetwork):
    def __init__(self, hyperparams: SqueezeNASNetCityscapesHyperparameters, cost_loss_multiplier: float, lr_aspp=True):
        super().__init__()
        self.hyperparams = hyperparams
        self.lr_aspp = lr_aspp
        self.cost_loss_multiplier = cost_loss_multiplier

        self.encoder = SuperNetworkSqueezeNASNet(
            hyperparams=hyperparams.to_ds_mobile_net_hyperparameters(last_channels=None, num_classes=None))

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
        cost = encoder_outputs['cost']

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
            return {'preds': logits, 'cost': cost}

        logits = logits.float()
        resource_cost_loss = torch.mean(self.cost_loss_multiplier * cost)
        problem_loss = self.criterion(logits, gt)
        loss = resource_cost_loss + problem_loss
        return {'loss': loss, 'preds': logits, 'cost': cost, 'problem_loss': problem_loss,
                'resource_cost_loss': resource_cost_loss}
