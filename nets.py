import torch

from arch.hyperparameters import get_cityscapes_hyperparams_small, get_cityscapes_hyperparams_large, \
    get_cityscapes_hyperparams_xlarge
from arch.model_cityscapes import SqueezeNASNetCityscapes
from arch.operations import Ops


def get_squeezenas_mac_small():
    out_ch = 19
    weight_path = "weights/mac_small.pth"
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.residual_skipish, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2]
    hyperparameters = get_cityscapes_hyperparams_small(1.0, out_ch, 16, 128, 7, 256)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_mac_large():
    out_ch = 19
    weight_path = "weights/mac_large.pth"
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g1_d2]
    hyperparameters = get_cityscapes_hyperparams_large(1.0, out_ch, 16, 128, 0.5, 8)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_mac_xlarge():
    out_ch = 19
    weight_path = "weights/mac_xlarge.pth"
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.residual_skipish, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k5_e6_g1, Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1_d2]
    hyperparameters = get_cityscapes_hyperparams_xlarge(1.0, out_ch, 48)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=False)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_small():
    out_ch = 19
    weight_path = "weights/lat_small.pth"
    genotype = [Ops.inverse_residual_k3_e1_g1, Ops.residual_skipish, Ops.residual_skipish, Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.residual_skipish, Ops.residual_skipish, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e3_g1_d2]
    hyperparameters = get_cityscapes_hyperparams_small(1.0, out_ch, 16, 128, 7, 256)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_large():
    out_ch = 19
    weight_path = "weights/lat_large.pth"
    genotype = [Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2]
    hyperparameters = get_cityscapes_hyperparams_large(1.0, out_ch, 16, 128, 0.5, 8)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_xlarge():
    out_ch = 19
    weight_path = "weights/lat_xlarge.pth"
    genotype = [Ops.residual_skipish, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e6_g1_d2]
    hyperparameters = get_cityscapes_hyperparams_xlarge(1.0, out_ch, 48)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=False)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


SQUEEZENAS_NETWORKS = {
    'squeezenas_mac_small': get_squeezenas_mac_small,
    'squeezenas_mac_large': get_squeezenas_mac_large,
    'squeezenas_mac_xlarge': get_squeezenas_mac_xlarge,
    'squeezenas_lat_small': get_squeezenas_lat_small,
    'squeezenas_lat_large': get_squeezenas_lat_large,
    'squeezenas_lat_xlarge': get_squeezenas_lat_xlarge
}
