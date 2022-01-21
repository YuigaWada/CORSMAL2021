import argparse
import numpy as np
import os
import csv
import torch

from task3.dataset import TestDataset, ValidationDataset, DebugDataset
from task3.video_processing import DynamicVideoProcessing
from task3.models import LoDE
from task3.utilities import calc_final_estimation

from task3.config import *


def run(args, csv_output_path):
    print("current device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    phase = 'test'
    output_path = 'outputs'  # todo: refactor
    if not os.path.exists(output_path): os.makedirs(output_path)

    dataset = ValidationDataset(args.path2data) if args.validation_test else TestDataset(args.path2data)
    video_processing = DynamicVideoProcessing(args, output_path, dataset=dataset)

    lode = LoDE(args, output_path, phase, video_processing, dataset=dataset)
    tag = "v1.0"

    lode.run(tag)

    # calculate final capacities

    if args.validation_test:
        scores, cscores = calc_final_estimation(dataset, average_training_set, phase, [tag])
        print("scores: {:.3f}".format(np.mean(scores)))

        for cid in sorted(list(cscores.keys())):
            print("{}: {:.3f}".format(cid, np.mean(cscores[cid])))
    else:
        pass  # todo: calc final
