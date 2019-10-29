import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from countmacs import MAC_Counter
from nets import SQUEEZENAS_NETWORKS

DATA_DIR = Path('./data')
os.environ['CITYSCAPES_DATASET'] = str(DATA_DIR.absolute())
print(os.environ['CITYSCAPES_DATASET'])

sys.path.insert(0, 'cityscapesScripts')  # add subdir to path

RESULTS_DIR = Path('./results')
DATASET_DIR = DATA_DIR / 'leftImg8bit/'
INPUT_DATA = list((DATASET_DIR / 'val').glob('**/*.png'))

normalize = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def map_back_to_35(gt):
    from cityscapesScripts.cityscapesscripts.helpers.labels import labels
    """ Maps neural net predictions back to 0-35 range, from 0-18 range """
    tmp = []
    for label in labels:
        if label.trainId != 255:
            tmp.append(label.id)  # creates a len 20 list [7,8,11,...]

    while len(tmp) <= 255:
        tmp.append(0)

    mapping = np.array(tmp)
    return mapping[gt]


def main(verbose: bool, only_macs: bool, save_output: bool = True, net_name: str = "squeezenas_mac_small",
         use_cpu: bool = True):
    result_dir = Path(RESULTS_DIR) / net_name
    result_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = result_dir / 'predictions'
    preds_dir.mkdir(exist_ok=True)
    os.environ['CITYSCAPES_RESULTS'] = str(preds_dir)

    from cityscapesScripts.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import \
        main as metrics  # noqa: E402

    net_constructor = SQUEEZENAS_NETWORKS[net_name]
    model = net_constructor()
    if not use_cpu:
        model = model.cuda()

    print('-' * 54)
    print(f'Evaluating {net_name}')
    print('-' * 54)

    print("Counting MACs")
    counter = MAC_Counter(model, [1, 3, 1024, 2048])
    if verbose:
        counter.print_layers()
    macs = counter.print_summary()

    print('-' * 54)

    if only_macs:
        return

    model.eval()

    print("Evaluating Model on the Validation Dataset")
    for idx, fname in enumerate(INPUT_DATA):  # run inference and save predictions
        print(f'\rSaving prediction {idx} out of {len(INPUT_DATA)}', end='')
        data = normalize(Image.open(fname))[None]
        if not use_cpu:
            data = data.cuda()
        output = model(data)
        pred = output['preds']
        pred = torch.argmax(pred[0], dim=0)
        pred = pred.cpu().data.numpy()
        pred = map_back_to_35(pred).astype(np.uint8)
        assert pred.shape == (1024, 2048), pred.shape
        preds_pil = Image.fromarray(pred, mode='L')
        preds_pil.save(preds_dir / fname.name, format='PNG')

    print('\n' + '-' * 54)
    sys.argv = [sys.argv[0]]
    metrics()  # run evaluation using Cityscapes metrics

    export_dir = result_dir / 'evaluationResults'

    if export_dir.exists():
        shutil.rmtree(export_dir)

    shutil.move(DATA_DIR / 'evaluationResults/', export_dir)
    with (result_dir / 'results.txt').open('w') as results_file:
        with open(export_dir / 'resultPixelLevelSemanticLabeling.json') as f:
            scores = json.load(f)  # get the net's performance
            classIOU = scores['averageScoreClasses']
            categoryIOU = scores['averageScoreCategories']

            line = f'Name: {net_name} \tclassIOU: {classIOU} \tcategoryIOU: {categoryIOU} \tGigaMACs: {macs["total_gmacs"]}\n'
            results_file.write(line)
            print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='if this option is supplied, a full layer'
                             ' by layer GigaMAC summary of the model will be printed. If this option is not supplied,'
                             ' only the total GigaMACs will be printed.')

    parser.add_argument('-m', '--only_macs', dest='only_macs', action='store_true',
                        help='if this option is supplied, no inference is run, only MAC count is printed.')

    parser.add_argument('-c', '--use_cpu', action='store_true',
                        help='If this option supplied, the network will be evaluated using the cpu.'
                             ' Otherwise the gpu will be used to run evaluation.')

    parser.add_argument('--net', type=str, choices=sorted(SQUEEZENAS_NETWORKS.keys()), default="squeezenas_lat_small")

    args, unknown = parser.parse_known_args()
    print(args)
    main(verbose=args.verbose, only_macs=args.only_macs, net_name=args.net, use_cpu=args.use_cpu)
