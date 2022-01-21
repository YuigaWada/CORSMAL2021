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

from utilities import create_initialized_row, list2csv


def run(args, csv_output_path):
    phase = 'test'
    output_path = 'outputs'  # todo: refactor
    if not os.path.exists(output_path): os.makedirs(output_path)

    dataset = ValidationDataset(args.path2data) if args.validation_task3 else TestDataset(args.path2data)
    video_processing = DynamicVideoProcessing(args, output_path, dataset=dataset)

    lode = LoDE(args, output_path, phase, video_processing, dataset=dataset)
    tag = "v1.0"

    result_list = []
    file_ids = dataset.get_all_fileids()
    for file_id in sorted(file_ids):
        if args.validation_task3 and int(file_id) not in dataset.set: continue  # isn't target

        # inference
        capacity = lode.run(file_id, tag)

        # save as dict
        arg_dict = create_initialized_row()
        arg_dict["Configuration ID"] = int(file_id)
        arg_dict["Container capacity"] = capacity
        result_list.append(arg_dict)

    list2csv(result_list, csv_output_path)
