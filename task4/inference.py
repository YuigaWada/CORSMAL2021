
import torch
import time
import numpy as np
import pandas as pd

from task3.dataset import TestDataset, ValidationDataset, DebugDataset, TrainDataset
from task3.video_processing import DynamicVideoProcessing
from task4.models import ConvNet
from task3.config import *
from task4.dataset import MaskDataset
from utilities import create_initialized_row, list2csv


def run(args, output_path, path_for_task3):
    test = TestDataset(args.path2data)
    df = pd.read_csv(path_for_task3)
    dataset = MaskDataset(test, df, is_test=True)

    model = ConvNet()
    model.load_state_dict(torch.load(args.task4_model_path))
    model.cuda()

    model.eval()
    result_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            print("[task4.inference] {} / {}".format(i, len(dataset)))
            (x_img, x_dimension_vector), _, idxs = dataset[i]
            fid = int(idxs[0])

            start_time = time.process_time()

            if x_img is None:
                pred = average_container_mass
            else:
                # inference
                pred = model(x_img.cuda().unsqueeze(0), x_dimension_vector.cuda().unsqueeze(0))
                pred = pred.detach().cpu().numpy()
                pred = pred.flatten()
                pred = pred[0] if pred.shape[0] > 0 else 0

            # measure time
            elapsed_time = time.process_time() - start_time
            # save as dict
            arg_dict = create_initialized_row()
            arg_dict["Configuration ID"] = fid
            arg_dict["Container mass"] = pred
            arg_dict["Execution time"] = elapsed_time * 1000 + dataset.get_elapsed_time(fid)
            result_list.append(arg_dict)

    list2csv(result_list, output_path)
