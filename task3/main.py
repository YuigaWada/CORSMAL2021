import argparse
import numpy as np
import os
import csv
import torch

from task3.dataset import ValidationDataset, DebugDataset
from task3.video_processing import DynamicVideoProcessing
from task3.models import LoDE
from task3.utilities import calc_final_estimation

from task3.config import *


def run(args, csv_output_path):
    print("current device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    phase = 'test'
    object_set = [str(i + 1) for i in range(9)]
    output_path = 'results'  # todo: refactor
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
    scores, cscores = calc_final_estimation(dataset, average_training_set, phase, [tag])
    print("scores: {:.3f}".format(np.mean(scores)))

    for cid in sorted(list(cscores.keys())):
        print("{}: {:.3f}".format(cid, np.mean(cscores[cid])))
