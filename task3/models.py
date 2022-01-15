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
from libs._3d.projection import *
from video_processing import AbstractVideoProcessing
from config import *


class LoDE:
    def __init__(self, args, output_path, phase, abstractVideoProcessing, dataset):
        self.args = args
        self.c = [dict.fromkeys(['rgb', 'seg', 'intrinsic', 'extrinsic']) for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.roi = [[] for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.output_path = output_path
        self.phase = phase
        self.video_processing = abstractVideoProcessing
        self.dataset = dataset

        # ADDED METHOD to extract the frames from the video
        print('Extract frames from bigger database')
        # for frame in frame_set:
        #     utilities.extract_frames(args.data_path, object_set, modality_set, frame)

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
            cv2.imwrite('{}/id{}_{}_{}.jpeg'.format(self.output_path, self.args.object, file_id, tag), visualization, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            plt.plot(radius)
            plt.savefig('{}/id{}_{}_{}_fig.png'.format(self.output_path, self.args.object, file_id, tag))
            plt.clf()
        except BaseException:
            capacity, height, width = -1, 0, 0

        return capacity, height, width

    def readData(self, fid, views, tag):
        c, roi = self.video_processing.prepare_data(self.detectionModel, self.args, fid, views, tag)
        for view in views:
            self.c[view]['rgb'] = c[view]['rgb']
            self.c[view]['seg'] = c[view]['seg']
            self.roi[view] = roi[view]

    # Read calibration file for the chosen setup

    def readCalibration(self, calibration_path, file_id):
        for view in range(1, VIEW_COUNT + 1):
            path = calibration_path + '/{}_c{}_calib.pickle'.format(file_id, view)

            if not os.path.exists(path):
                print('COMBINATION OF PARAMETERS FOR CALIBRATION DOES NOT EXISTS')
                return
            else:
                with open(path, 'rb') as f:
                    calibration = pickle.load(f, encoding="latin1")
                    c1_intrinsic = calibration[0]
                    c1_extrinsic = calibration[1]

            self.c[view]['intrinsic'] = c1_intrinsic['rgb']
            self.c[view]['extrinsic'] = c1_extrinsic['rgb']

    def run(self, tag):
        calibration_path = os.path.join(self.args.data_path, self.args.object, 'calib')
        assert os.path.isdir(calibration_path), "Can't find path " + calibration_path

        file_pattern = r"([\w\d_]+)_c1_calib.pickle"
        file_id_list = [re.match(file_pattern, f).group(1) for f in os.listdir(calibration_path) if re.match(file_pattern, f)]

        for fid in sorted(file_id_list):
            if int(fid) not in self.dataset.dict[int(self.args.object)]: continue  # isn't target
            # Read camera calibration files
            self.readCalibration(calibration_path, fid)
            # Main loop
            self.readData(fid, views=[1, 2], tag=tag)

            answer = float(self.dataset.annotations[int(fid)]["container capacity"])
            capacity, height, width = self.getObjectDimensions(fid, self.c[1], self.c[2], self.roi[1], self.roi[2], tag)

            if capacity == -1:  # 失敗したらview1-view3間で再度実行
                self.readData(fid, views=[1, 2], tag=tag)
                capacity, height, width = self.getObjectDimensions(fid, self.c[1], self.c[3], self.roi[1], self.roi[3], tag)
                if capacity == -1:
                    print('Error measuring id{}_{}'.format(self.args.object, fid))
                    capacity = average_training_set
                else:
                    print('{}/id{}_{} ---- DONE (view1-view3)'.format(self.output_path, self.args.object, fid))
            else:
                print('{}/id{}_{} ---- DONE (view1-view2)'.format(self.output_path, self.args.object, fid))

            diff = abs(answer - capacity)
            score = np.exp(-diff / answer)

            with open('{}/estimation_{}_{}.csv'.format(self.output_path, tag, self.phase), 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id{}_{}.png'.format(self.args.object, fid), height, width, capacity, tag, diff, score])
