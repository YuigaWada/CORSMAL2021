
import torch
import torchvision
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
    video_processing = DynamicVideoProcessing(output_path='outputs', dataset=test)  # todo: MaskDataset内で生成するように

    detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    detectionModel.eval()
    detectionModel.cuda()

    df = pd.read_csv(path_for_task3)
    X = test.get_all_fileids()
    X = list(map(int, X))
    dataset = MaskDataset(X, np.empty_like(X), df, video_processing, detectionModel, args)

    model = ConvNet()
    model.load_state_dict(torch.load('./task4.pt'))  # todo: 指定できるように
    model.cuda()

    model.eval()
    result_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            (x_img, x_dimension_vector), _, idxs = dataset[i]

            # inference
            pred = model(x_img.cuda().unsqueeze(0), x_dimension_vector.cuda().unsqueeze(0))
            pred = pred.detach().cpu().numpy()
            pred = pred.flatten()
            pred = pred[0] if pred.shape[0] > 0 else 0

            # save as dict
            arg_dict = create_initialized_row()
            arg_dict["Configuration ID"] = int(idxs[0])
            arg_dict["Container mass"] = pred

            result_list.append(arg_dict)

    list2csv(result_list, output_path)
