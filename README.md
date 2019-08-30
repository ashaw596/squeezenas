# SqueezeNAS
Model Repo for the [paper](https://arxiv.org/abs/1908.01748) ```SqueezeNAS: Fast neural architecture search for faster semantic segmentation```.

Includes the trained models and architectures as well as evaluation code.

## Requirements
```
Python >= 3.7.0, pytorch >= 1.0.1, torchvision >= 0.2.2, tabulate == 0.8.3, numpy >= 1.15.4
```
## Cityscapes Dataset
The dataset can be downloaded from their [website](https://www.cityscapes-dataset.com/downloads/).

Extract the [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and copy the ```gtFine``` directory to ```data/gtFine```.  
Extract the [left imaget set](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and copy the ```leftImg8bit``` directory to ```data/leftImg8bit```.

## Instructions
Install the prerequesites. The evaluation will be faster on GPU, but it can also be evaluated without using the ```--use_cpu``` flag. Install the corresponding version of [pytorch](https://pytorch.org)

Clone the repository.

Initialize the cityscapesScripts submodule.
```
git submodule init
git submodule update
```
Alternatively download a [zip](https://github.com/mcordts/cityscapesScripts/archive/master.zip) of the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts/) repo and extract the ```cityscapesscripts``` folder to
```cityscapesScripts/cityscapesscripts```

Download and extract the Cityscapes dataset to the data folder as described above.

## Evaluating

Use ```eval.py``` to evalute the models.

There is some minor variation in results reported in the paper which is the maximum validation accuracy chosen from the last 20 epochs of training due to the models which were saved.

To evaluate the squeezenas_lat_small network:  
```python3 eval.py --net squeezenas_lat_small```  
Expected results:  
```classIOU: 0.6800969532611763, categoryIOU: 0.8413112105833828, GigaMACs: 4.465098752```

To evaluate the squeezenas_lat_large network:  
```python3 eval.py --net squeezenas_lat_large```  
Expected results:  
```classIOU: 0.7350594481447746, categoryIOU: 0.8657799839081778, GigaMACs: 19.574636544```

To evaluate the squeezenas_lat_xlarge network:  
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
```classIOU: 0.7462089926605817, categoryIOU: 0.8699823279398887, GigaMACs: 21.838299135999996```

## eval.py command line options

```eval.py``` uses a command line interface. 
For help, please run: ```python3 eval.py --help```

The following flags are available:  
-v, --verbose: if this option is supplied, a full layer by layer GMAC summary of the model will be printed. If this option is not supplied, only the total GMACs will be printed.

-m, --only_macs: if this option is supplied, no inference is run, only MAC count is printed.

-n, --nets: this allows choice from {squeezenas_lat_large,squeezenas_lat_small,squeezenas_lat_xlarge,squeezenas_mac_large,squeezenas_mac_small,squeezenas_mac_xlarge}, indicating which net to evaluate.

-c, --use_cpu: If this option supplied, the network will be evaluated using the cpu. Otherwise the gpu will be used to run
                        evaluation.
