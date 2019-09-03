import torch

from arch.hyperparameters import get_cityscapes_hyperparams_small, get_cityscapes_hyperparams_large, \
    get_cityscapes_hyperparams_xlarge
from arch.model_cityscapes import SqueezeNASNetCityscapes
from arch.operations import Ops


def get_squeezenas_mac_small():
    # noinspection PyPep8
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.residual_skipish, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2]
    weight_path = "weights/mac_small.pth"
    hyperparameters = get_cityscapes_hyperparams_small()
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_mac_large():
    # noinspection PyPep8
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g1_d2]
    weight_path = "weights/mac_large.pth"
    hyperparameters = get_cityscapes_hyperparams_large()
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_mac_xlarge():
    # noinspection PyPep8
    genotype = [Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.residual_skipish, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k5_e6_g1, Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1_d2]
    weight_path = "weights/mac_xlarge.pth"
    hyperparameters = get_cityscapes_hyperparams_xlarge()
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=False)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_small():
    # noinspection PyPep8
    genotype = [Ops.inverse_residual_k3_e1_g1, Ops.residual_skipish, Ops.residual_skipish, Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.residual_skipish, Ops.residual_skipish, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e1_g2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e3_g1_d2]
    weight_path = "weights/lat_small.pth"
    hyperparameters = get_cityscapes_hyperparams_small()
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_large():
    # noinspection PyPep8
    genotype = [Ops.residual_skipish, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e1_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e6_g1_d2]
    weight_path = "weights/lat_large.pth"
    hyperparameters = get_cityscapes_hyperparams_large()
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_xlarge():
    # noinspection PyPep8
    genotype = [Ops.residual_skipish, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e1_g1_d2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k5_e1_g2, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e6_g1, Ops.inverse_residual_k5_e6_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e1_g2_d2, Ops.inverse_residual_k3_e3_g1, Ops.inverse_residual_k3_e6_g1_d2, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k5_e1_g1, Ops.inverse_residual_k3_e3_g1_d2, Ops.inverse_residual_k3_e6_g1_d2]
    weight_path = "weights/lat_xlarge.pth"
    hyperparameters = get_cityscapes_hyperparams_xlarge()
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
