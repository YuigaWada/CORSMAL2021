##################################################################################
# This work is an extension of LoDE code developed by Ricardo Sanchez Matilla
# (Email: ricardo.sanchezmatilla@qmul.ac.uk)
#        Author: Francesca Palermo
#         Email: f.palermo@qmul.ac.uk
#         Date: 2020/09/03
# Centre for Intelligent Sensing, Queen Mary University of London, UK
#
##################################################################################
# License
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##################################################################################
# System libs

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision

import os

import pickle
import re
import csv
import matplotlib.pyplot as plt

from task3.libs._3d.projection import *
from task3.video_processing import AbstractVideoProcessing
from task3.config import *


class LoDE:
    def __init__(self, args, output_path, phase, abstractVideoProcessing, dataset):
        self.args = args
        self.c = [dict.fromkeys(['rgb', 'seg', 'intrinsic', 'extrinsic']) for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.roi = [[] for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.output_path = output_path
        self.phase = phase
        self.video_processing = abstractVideoProcessing
        self.dataset = dataset

        # Load object detection model
        self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.detectionModel.eval()
        self.detectionModel.cuda()

    def getObjectDimensions(self, file_id, c1, c2, roi_list1, roi_list2, tag):
        try:
            centroid1, contour1 = getCentroid(c1['seg'])
            centroid2, contour2 = getCentroid(c2['seg'])

            centroid = cv2.triangulatePoints(c1['extrinsic']['projMatrix'], c2['extrinsic']['projMatrix'],
                                             centroid1, centroid2).transpose()
            centroid /= centroid[:, -1].reshape(-1, 1)
            centroid = centroid[:, :-1].reshape(-1)

            height, width, visualization, capacity, radius = getObjectDimensions(c1, c2, roi_list1, roi_list2, centroid, self.args.draw)
            # cv2.imwrite('{}/id{}_{}_{}.jpeg'.format(self.output_path, self.args.object, file_id, tag), visualization, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            # plt.plot(radius)
            # plt.savefig('{}/id{}__{}_fig.png'.format(self.output_path, file_id, tag))
            # plt.clf()
        except BaseException:
            capacity, height, width = -1, -1, (-1, -1)

        return capacity, height, width

    def readData(self, fid, views, tag):
        c, roi = self.video_processing.prepare_data(self.detectionModel, self.args, fid, views, tag)
        for view in views:
            self.c[view]['rgb'] = c[view]['rgb']
            self.c[view]['seg'] = c[view]['seg']
            self.roi[view] = roi[view]

    def readCalibration(self, file_id):
        for view in range(1, VIEW_COUNT + 1):
            calibration_path = self.dataset.get_calib_path(file_id, view)
            if not os.path.exists(calibration_path):
                print('failed to load {}'.format(calibration_path))
                return
            else:
                with open(calibration_path, 'rb') as f:
                    calibration = pickle.load(f, encoding="latin1")
                    c1_intrinsic = calibration[0]
                    c1_extrinsic = calibration[1]

            self.c[view]['intrinsic'] = c1_intrinsic['rgb']
            self.c[view]['extrinsic'] = c1_extrinsic['rgb']

    def run(self, fid, tag):
        # Read camera calibration files
        self.readCalibration(fid)
        self.readData(fid, views=[1, 2], tag=tag)

        capacity, height, width = self.getObjectDimensions(fid, self.c[1], self.c[2], self.roi[1], self.roi[2], tag)

        if capacity == -1:  # 失敗したらview1-view3間で再度実行
            self.readData(fid, views=[1, 3], tag=tag)
            capacity, height, width = self.getObjectDimensions(fid, self.c[1], self.c[3], self.roi[1], self.roi[3], tag)
            if capacity == -1:
                print('Error measuring id{}'.format(fid))
                capacity = average_training_set
            else:
                print('{}/id{} ---- DONE (view1-view3)'.format(self.output_path, fid))
        else:
            print('{}/id{} ---- DONE (view1-view2)'.format(self.output_path, fid))

        return capacity, height, width
