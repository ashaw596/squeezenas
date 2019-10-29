from typing import Sequence

from torch import nn

from arch.model import InverseResidualMetaNetHyperparameters
from arch.operations import Ops
from arch.utils import conv3x3_bn, conv_1x1_bn, Flatten
from search.arch_search import MixedModule, SuperNetwork

SQUEEZENAS_SEARCH_SPACE = (Ops.mobile_net_k3_e1_g1,
                           Ops.mobile_net_k3_e1_g2,
                           Ops.mobile_net_k3_e3_g1,
                           Ops.mobile_net_k3_e6_g1,
                           Ops.mobile_net_k5_e1_g1,
                           Ops.mobile_net_k5_e1_g2,
                           Ops.mobile_net_k5_e3_g1,
                           Ops.mobile_net_k5_e6_g1,
                           Ops.mobile_net_k3_e1_g1_d2,
                           Ops.mobile_net_k3_e1_g2_d2,
                           Ops.mobile_net_k3_e3_g1_d2,
                           Ops.mobile_net_k3_e6_g1_d2,
                           Ops.mobile_net_residual_skipish,)


class MixedInvertedResidual(MixedModule):
    def __init__(self, c_in: int, c_out: int, stride: int, ops: Sequence[Ops]):
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and c_in == c_out
        super().__init__([(op.name, op.value(c_in, c_out, stride, affine=True)) for op in ops])

    def forward(self, x):
        output, cost = super().forward(x)
        if self.use_res_connect:
            return x + output, cost
        else:
            return output, cost


class SuperNetworkSqueezeNASNet(SuperNetwork):
    def __init__(self, hyperparams: InverseResidualMetaNetHyperparameters, dropout=0, ops=SQUEEZENAS_SEARCH_SPACE):
        super().__init__()
        self.hyperparams = hyperparams
        self.conv1 = conv3x3_bn(c_in=3, c_out=hyperparams.init_channels, stride=2)
        self.residuals: nn.ModuleList[MixedInvertedResidual] = nn.ModuleList()

        gene_i = 0
        last_c = hyperparams.init_channels

        for block_hyper in hyperparams.blocks:
            for i in range(block_hyper.num_repeat):
                if i == 0:
                    stride = block_hyper.stride
                else:
                    stride = 1
                self.residuals.append(MixedInvertedResidual(last_c, block_hyper.num_channels, stride, ops))
                gene_i += 1
                last_c = block_hyper.num_channels

        if hyperparams.last_channels is not None:
            self.last_conv = conv_1x1_bn(last_c, hyperparams.last_channels)
            if hyperparams.num_classes is not None:
                # building classifier
                if hyperparams.last_pooled_channels is None:
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        Flatten(),
                        nn.Dropout(dropout),
                        nn.Linear(hyperparams.last_channels, hyperparams.num_classes),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        Flatten(),
                        nn.Linear(hyperparams.last_channels, hyperparams.last_pooled_channels),
                        nn.Dropout(dropout),
                        nn.Linear(hyperparams.last_pooled_channels, hyperparams.num_classes),
                    )
                self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, gt=None):
        total_cost = 0
        cur_feat = inputs
        cur_feat = self.conv1(cur_feat)
        residuals_outputs = []
        for i, residual in enumerate(self.residuals):
            cur_feat, cost = residual(cur_feat)
            residuals_outputs.append(cur_feat)
            total_cost += cost

        if self.hyperparams.last_channels is None:
            return {'output': cur_feat, 'residuals_outputs': residuals_outputs, 'cost': total_cost}

        cur_feat = self.last_conv(cur_feat)

        if self.hyperparams.num_classes is None:
            return {'output': cur_feat, 'residuals_outputs': residuals_outputs, 'cost': total_cost}

        logits = self.classifier(cur_feat)

        if gt is None:
            return {'preds': logits, 'cost': total_cost}
        else:
            loss = self.criterion(logits.float(), gt)
            return {'loss': loss, 'preds': logits, 'cost': total_cost}
