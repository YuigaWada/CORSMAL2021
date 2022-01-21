import argparse
import numpy as np
import os
import csv
import utilities
import torch

from dataset import ValidationDataset, DebugDataset
from video_processing import DynamicVideoProcessing
from models import LoDE

from config import *

def run(args):
    print("current device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    phase = 'test'
    object_set = [str(i + 1) for i in range(9)]
    output_path = 'results'
    if not os.path.exists(output_path): os.makedirs(output_path)

    dataset = ValidationDataset(args.path2data)
    video_processing = DynamicVideoProcessing(args, output_path)

    print(f'Executing on {object_set} containers...')
    lode = LoDE(args, output_path, phase, video_processing, dataset=dataset)
    output_path = 'results'
    tag = "v1.0"
    with open('{}/estimation_{}_{}.csv'.format(output_path, tag, phase), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fileName', 'height[mm]', 'width[mm]', 'capacity[mL]', 'frame', 'capacity_diff[mL]', 'score'])

    for args.object in object_set:
        lode.run(tag=tag)

    # calculate final capacities
    scores, cscores = utilities.calc_final_estimation(dataset, average_training_set, phase, [tag])
    print("scores: {:.3f}".format(np.mean(scores)))

    for cid in sorted(list(cscores.keys())):
        print("{}: {:.3f}".format(cid, np.mean(cscores[cid])))

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str)
    parser.add_argument("output_path", help="Path to output the resulting csv file e.g.) ./out.csv", type=str)
    parser.add_argument('--draw', default=True, action='store_true', help='Output visual results in ./results')
    parser.add_argument('--validation_test', default=True, action='store_true', help='Output visual results in ./results') # todo: チェック
    parser.add_argument( 
        "-m",
        "--model_path",
        help='Path to the stored model. Defaults to "./task_challenger3.pt"',
        default="./task_challenger3.pt",
    )

    args = parser.parse_args()
    run(args)
