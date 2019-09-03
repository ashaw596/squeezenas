import dataclasses

from arch.model import InvertResidualNetBlockMetaHyperparameters
from arch.model_cityscapes import SqueezeNASNetCityscapesHyperparameters


def get_cityscapes_hyperparams_small(width_multiplier=1.0, num_classes=19, init_channels=16, mid_channels=128,
                                     skip_output_block_index=7, last_conv_channels=256):
    block_hyperparams = (InvertResidualNetBlockMetaHyperparameters(num_channels=16, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=24, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=40, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=48, num_repeat=4, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=96, num_repeat=4, stride=1),)
    if width_multiplier != 1.0:
        block_hyperparams = tuple(
            dataclasses.replace(meta_block, num_channels=round(meta_block.num_channels * width_multiplier))
            for meta_block in block_hyperparams)
    hyperparams = SqueezeNASNetCityscapesHyperparameters(init_channels=init_channels, blocks=block_hyperparams,
                                                         num_classes=num_classes,
                                                         skip_output_block_index=skip_output_block_index,
                                                         mid_channels=mid_channels, last_channels=last_conv_channels)
    return hyperparams


def get_cityscapes_hyperparams_large(width_multiplier=1.0, num_classes=19, init_channels=16, mid_channels=128,
                                     final_width_multiplier=0.5, skip_output_block_index=8):
    block_hyperparams = (InvertResidualNetBlockMetaHyperparameters(num_channels=16, num_repeat=1, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=24, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=32, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=64, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=96, num_repeat=4, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=160, num_repeat=4, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=round(320 * final_width_multiplier),
                                                                   num_repeat=1, stride=1),)
    if width_multiplier != 1.0:
        block_hyperparams = tuple(
            dataclasses.replace(meta_block, num_channels=round(meta_block.num_channels * width_multiplier))
            for meta_block in block_hyperparams)
    hyperparams = SqueezeNASNetCityscapesHyperparameters(init_channels=init_channels, blocks=block_hyperparams,
                                                         num_classes=num_classes,
                                                         skip_output_block_index=skip_output_block_index,
                                                         mid_channels=mid_channels)
    return hyperparams


def get_cityscapes_hyperparams_xlarge(width_multiplier=1.0, num_classes=19, init_channels=48, mid_channels=160,
                                      final_width_multiplier=1.0, skip_output_block_index=4):
    block_hyperparams = (InvertResidualNetBlockMetaHyperparameters(num_channels=24, num_repeat=1, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=32, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=48, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=96, num_repeat=4, stride=2),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=144, num_repeat=4, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=240, num_repeat=4, stride=1),
                         InvertResidualNetBlockMetaHyperparameters(num_channels=round(320 * final_width_multiplier),
                                                                   num_repeat=1, stride=1),)
    if width_multiplier != 1.0:
        block_hyperparams = tuple(
            dataclasses.replace(meta_block, num_channels=round(meta_block.num_channels * width_multiplier))
            for meta_block in block_hyperparams)
    hyperparams = SqueezeNASNetCityscapesHyperparameters(init_channels=init_channels, blocks=block_hyperparams,
                                                         num_classes=num_classes,
                                                         skip_output_block_index=skip_output_block_index,
                                                         mid_channels=mid_channels)
    return hyperparams
