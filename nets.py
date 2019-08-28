import torch

from arch.operations import Ops

from arch.hyperparameters import get_small_cityscapes_hyperparams_v2, get_mobilenet_cityscapes_hyperparams, \
    get_mobilenet_cityscapes_hyperparams_v2
from arch.model_mobilenet_cityscapes import DSMobileNetMetaCityscapes


def get_squeezenas_mac_small():
    out_ch = 19
    weight_path = "weights/mac_small.pth"
    genotype = [Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_residual_skipish, Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k3_e1_g1_d2, Ops.mobile_net_k3_e1_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2_d2]
    hyperparameters = get_small_cityscapes_hyperparams_v2(1.0,out_ch,16,128,7,256)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_squeezenas_mac_large():
    out_ch = 19
    weight_path = "weights/mac_large.pth"
    genotype = [Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k3_e6_g1, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e1_g1, Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k5_e1_g1, Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g1_d2]
    hyperparameters = get_mobilenet_cityscapes_hyperparams(1.0,out_ch,16,128,0.5,8)#get_mobilenet_cityscapes_hyperparams(0.75,,16,128,final_width_multiplier=0.375,skip_output_block_index=8)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_mac_xlarge():
    out_ch = 19
    weight_path = "weights/mac_xlarge.pth"
    genotype = [Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k3_e1_g2, Ops.mobile_net_k5_e1_g1,
                Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e1_g2,
                Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e3_g1,
                Ops.mobile_net_residual_skipish, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k5_e6_g1,
                Ops.mobile_net_k3_e1_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_residual_skipish,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e1_g2_d2,
                Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e3_g1_d2]
    hyperparameters = get_mobilenet_cityscapes_hyperparams_v2(1.0, out_ch, 48)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=False)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_small():
    out_ch = 19
    weight_path = "weights/lat_small.pth"
    genotype = [Ops.mobile_net_k3_e1_g1, Ops.mobile_net_residual_skipish, Ops.mobile_net_residual_skipish,
                Ops.mobile_net_residual_skipish, Ops.mobile_net_k3_e6_g1, Ops.mobile_net_k3_e6_g1,
                Ops.mobile_net_residual_skipish, Ops.mobile_net_residual_skipish, Ops.mobile_net_k5_e6_g1,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k5_e1_g2, Ops.mobile_net_k3_e1_g2,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2,
                Ops.mobile_net_k3_e1_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e3_g1_d2,
                Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k3_e3_g1_d2]
    hyperparameters = get_small_cityscapes_hyperparams_v2(1.0, out_ch, 16, 128, 7, 256)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_squeezenas_lat_large():
    out_ch = 19
    weight_path = "weights/lat_large.pth"
    genotype = [Ops.mobile_net_residual_skipish, Ops.mobile_net_k3_e6_g1, Ops.mobile_net_k3_e3_g1,
                Ops.mobile_net_k3_e1_g1_d2, Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e6_g1,
                Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e1_g1,
                Ops.mobile_net_k5_e1_g1, Ops.mobile_net_k5_e1_g2, Ops.mobile_net_k3_e6_g1_d2,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e1_g1,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e6_g1_d2]
    hyperparameters = get_mobilenet_cityscapes_hyperparams(1.0, out_ch, 16, 128, 0.5, 8)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=True)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_squeezenas_lat_xlarge():
    out_ch = 19
    weight_path = "weights/lat_xlarge.pth"
    genotype = [Ops.mobile_net_residual_skipish, Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k3_e3_g1,
                Ops.mobile_net_k3_e3_g1, Ops.mobile_net_k3_e1_g1_d2, Ops.mobile_net_k3_e6_g1, Ops.mobile_net_k3_e3_g1,
                Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k5_e1_g2, Ops.mobile_net_k5_e6_g1, Ops.mobile_net_k3_e6_g1,
                Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e6_g1, Ops.mobile_net_k5_e6_g1,
                Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k3_e1_g2_d2, Ops.mobile_net_k3_e3_g1,
                Ops.mobile_net_k3_e6_g1_d2, Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k5_e1_g1,
                Ops.mobile_net_k3_e3_g1_d2, Ops.mobile_net_k3_e6_g1_d2]
    hyperparameters = get_mobilenet_cityscapes_hyperparams_v2(1.0, out_ch, 48)
    model = DSMobileNetMetaCityscapes(hyperparameters, genotype, lr_aspp=False)
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
