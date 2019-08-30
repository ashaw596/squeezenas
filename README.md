# SqueezeNAS: Fast neural architecture search for faster semantic segmentation

This repository contains the trained models and architectures as well as evaluation
code for the [SqueezeNAS paper](https://arxiv.org/abs/1908.01748).

## Requirements

```
Python >= 3.7.0
PyTorch >= 1.0.1
torchvision >= 0.2.2,
tabulate == 0.8.3,
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

(Alternatively download a [zip](https://github.com/mcordts/cityscapesScripts/archive/master.zip) of the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts/) repo and extract the `cityscapesscripts` folder to
`cityscapesScripts/cityscapesscripts`.)

## Evaluation

Expected results:

| Name | classIOU  | categoryIOU | GigaMACs |
|---|---|---|---|---|---|
| squeezenas_lat_small | 0.6800 | 0.8413 | 4.46 |
| squeezenas_lat_large | 0.7350 | 0.8657 | 10.86 |
<!-- |   |   |   |   |   |   | -->


<!-- To evaluate the squeezenas_lat_xlarge network:
```python3 eval.py --net squeezenas_lat_xlarge```
Expected Results:
```classIOU: 0.7505558962490358, categoryIOU: 0.8727507286930606, GigaMACs: 32.729202687999994```

To evaluate the squeezenas_mac_small network:
```python3 eval.py --net squeezenas_mac_small```
Expected results:
```classIOU: 0.6677272073464366, categoryIOU: 0.8393623681041512, GigaMACs: 3.007512576000001```

To evaluate the squeezenas_mac_large network:
```python3 eval.py --net squeezenas_mac_large```
Expected Results:
```classIOU: 0.7240103001347041, categoryIOU: 0.8617861866154775, GigaMACs: 9.386147840000001```

To evaluate the squeezenas_mac_xlarge network:
```python3 eval.py --net squeezenas_mac_xlarge```
Expected Results:
```classIOU: 0.7462089926605817, categoryIOU: 0.8699823279398887, GigaMACs: 21.838299135999996``` -->

## eval.py command line options

```eval.py``` uses a command line interface.
For help, please run: ```python3 eval.py --help```


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
--net {squeezenas_lat_large,squeezenas_lat_small,squeezenas_lat_xlarge,squeezenas_mac_large,squeezenas_mac_small,squeezenas_mac_xlarge}
```
