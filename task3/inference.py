import time

from task3.dataset import TestDataset, ValidationDataset, DebugDataset, TrainDataset
from task3.video_processing import DynamicVideoProcessing
from task3.models import LoDE
from task3.config import *

from utilities import create_initialized_row, list2csv


def run(args, csv_output_path):
    phase = 'test'
    output_path = 'outputs'

    dataset = ValidationDataset(args.path2data, ratio=0.001) if args.validation_task3 else TestDataset(args.path2data)
    video_processing = DynamicVideoProcessing(output_path='outputs', dataset=dataset)

    lode = LoDE(args, output_path, phase, video_processing, dataset=dataset)
    tag = "v1.0"

    result_list = []
    file_ids = dataset.get_all_fileids()
    count = 0
    for file_id in sorted(file_ids):
        if args.validation_task3 and int(file_id) not in dataset.set: continue  # isn't target

        print("[task3.inference] {} / {}".format(count, len(file_ids)))
        start_time = time.process_time()

        # inference
        capacity, height, (width_top, width_bottom) = lode.run(file_id, tag)

        # measure time
        elapsed_time = time.process_time() - start_time

        # save as dict
        arg_dict = create_initialized_row()
        arg_dict["Configuration ID"] = int(file_id)
        arg_dict["Container capacity"] = capacity
        arg_dict["Height"] = height
        arg_dict["Width at the top"] = width_top
        arg_dict["Width at the bottom"] = width_bottom
        arg_dict["Execution time"] = elapsed_time
        result_list.append(arg_dict)
        count += 1

    list2csv(result_list, csv_output_path)
