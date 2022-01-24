import numpy as np

from task3.dataset import TestDataset, ValidationDataset, DebugDataset, TrainDataset
from task3.video_processing import DynamicVideoProcessing
from task4.models import ConvNet
from task3.config import *
from task4.dataset import MaskDataset
from task4.utilities import EarlyStopping

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


def generate_dataset(train, validation, video_processing, detectionModel, args, batch_size, path_for_task3):
    X_train, y_train = train.get_container_mass_data()
    X_test, y_test = validation.get_container_mass_data()
    df = pd.read_csv(path_for_task3)

    train_dataset = MaskDataset(X_train, y_train, df, video_processing, detectionModel, args)
    test_dataset = MaskDataset(X_test, y_test, df, video_processing, detectionModel, args, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0)

    return train_loader, test_loader


def train(args, path_for_task3):
    train = TrainDataset(args.path2data, ratio=0.8)
    validation = ValidationDataset(args.path2data, ratio=0.2)
    video_processing = DynamicVideoProcessing(output_path='outputs', dataset=train)  # todo: MaskDataset内で生成するように

    detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    detectionModel.eval()
    detectionModel.cuda()

    train_loader, val_loader = generate_dataset(train, validation, video_processing, detectionModel, args, batch_size=8, path_for_task3=path_for_task3)

    model = ConvNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(model)

    EPOCH = 5000
    has_saved = False
    for epoch in range(EPOCH):
        total_loss = 0.0
        model.train()
        for _, batch in enumerate(train_loader):
            (x_img, x_dimension_vector), y = batch[0], batch[1].cuda()
            x_img, x_dimension_vector = x_img.cuda(), x_dimension_vector.cuda()

            optimizer.zero_grad()

            pred = model(x_img, x_dimension_vector)

            loss = criterion(pred, y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            score = eval(validation, val_loader, model=model)
            print("validation score:", score)

            should_stop = early_stopping.update(total_loss)
            if should_stop:
                print("==============")
                print("final result")
                print("--------------")

                score = eval(validation, val_loader, model=None)
                print("epoch:", epoch)
                print("score:", score)

                print("==============")

                has_saved = True
                break

        info = "loss = {:.3f}".format(total_loss)
        print("Epoch {} / {} ({})".format(epoch + 1, EPOCH, info))

    if not has_saved:
        torch.save(model.state_dict(), './task4.pt')

    return model


def eval(validation, val_loader, model=None):
    if model is None:
        model = ConvNet()
        model.load_state_dict(torch.load('./task4.pt'))  # todo: 指定できるように
        model.cuda()
    scores = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            (x_img, x_dimension_vector), _, idxs = batch[0], batch[1].cuda(), batch[2].cuda()
            x_img, x_dimension_vector = x_img.cuda(), x_dimension_vector.cuda()

            preds = model(x_img, x_dimension_vector)
            for i in range(preds.shape[0]):
                idx = idxs[i].detach().cpu().numpy().astype(np.uint32)
                pred = preds[i].detach().cpu().numpy()
                answer = validation.annotations[idx[0]]["container mass"]
                if batch_idx == 0 and i == 0: print(pred, answer)
                score = np.exp(-np.abs(pred - answer) / answer)
                scores.append(score)

    total_score = np.mean(scores)
    return total_score
