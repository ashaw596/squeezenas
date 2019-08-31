# SqueezeNAS: Fast neural architecture search for faster semantic segmentation

This repository contains the trained models and architectures as well as evaluation
code for the [SqueezeNAS paper](https://arxiv.org/abs/1908.01748).

If you find this useful, or if you use it in your commercial work, please cite:

    @inproceedings{2019_SqueezeNAS,
        author = {Albert Shaw and Daniel Hunter and Forrest Iandola and Sammy Sidhu},
        title = {{SqueezeNAS}: Fast neural architecture search for faster semantic segmentation},
        booktitle = {ICCV Neural Architects Workshop},
        year = {2019}
    }

## Requirements

```
Python >= 3.7.0
PyTorch >= 1.0.1
torchvision >= 0.2.2
tabulate == 0.8.3
numpy >= 1.15.4
Pillow
```

## Cityscapes Dataset

The dataset can be downloaded from [the cityscapes website](https://www.cityscapes-dataset.com/downloads/).

1. Extract the [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
and copy the `gtFine` directory to `data/gtFine`.
1. Extract the [left camera images](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
and copy the `leftImg8bit` directory to `data/leftImg8bit`.

## Instructions

1. Install the required packages.
1. Clone this repository.
1. Download and extract the Cityscapes dataset to the data folder as described above.
1. Initialize the cityscapesScripts submodule: `git submodule init && git submodule update`.

(Alternatively download a [zip](https://github.com/mcordts/cityscapesScripts/archive/master.zip) of the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts/) repo and extract the `cityscapesscripts` folder to `cityscapesScripts/cityscapesscripts`.)

## Evaluation

Expected results:

| Name | classIOU  | categoryIOU | GigaMACs |
| --- | --- | --- | --- |
| squeezenas_lat_small  | 0.6800 | 0.8413 | 4.46  |
| squeezenas_lat_large  | 0.7350 | 0.8657 | 10.86 |
| squeezenas_lat_xlarge | 0.7505 | 0.8727 | 32.72 |
| squeezenas_mac_small  | 0.6677 | 0.8393 | 3.00  |
| squeezenas_lat_large  | 0.7240 | 0.8617 | 9.38  |
| squeezenas_mac_xlarge | 0.7462 | 0.8699 | 21.83 |

## eval.py command line options

`eval.py` uses a command line interface.
For help, please run: `python3 eval.py --help`

The following flags are available:
```
-h, --help            show this help message and exit
-v, --verbose         if this option is supplied, a full layer by layer
                      GigaMAC summary of the model will be printed. If this
                      option is not supplied, only the total GigaMACs will
                      be printed.
-m, --only_macs       if this option is supplied, no inference is run, only
                      MAC count is printed.
-c, --use_cpu         If this option supplied, the network will be evaluated
                      using the cpu. Otherwise the gpu will be used to run
                      evaluation.
--net {squeezenas_lat_large,squeezenas_lat_small,squeezenas_lat_xlarge,squeezenas_mac_large,
  squeezenas_mac_small,squeezenas_mac_xlarge}
```
