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

#
# OpenCV
import cv2
#
# Numpy
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
from numpy.linalg import inv
#
import pickle
#
# ScyPy
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
#
import math
#
import copy
#
from task3.utilities import draw_bbox, roi_extract
#


def getCentroid(mask):
    # Get the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Get centroid of the largest contour
    M = cv2.moments(largest_contour)

    try:
        centroid = np.array((M['m10'] / M['m00'], M['m01'] / M['m00']))
        return centroid, largest_contour.squeeze()
    except BaseException:
        print('Centroid not found')
        return None, None

#


def triangulate(c1, c2, point1, point2, undistort=True):

    if (point1.dtype != 'float64'):
        point1 = point1.astype(np.float64)

    if (point2.dtype != 'float64'):
        point2 = point2.astype(np.float64)

    point3d = cv2.triangulatePoints(c1.extrinsic['rgb']['projMatrix'], c2.extrinsic['rgb']['projMatrix'], point1.reshape(2, 1), point2.reshape(2, 1)).transpose()
    for point in point3d:
        point /= point[-1]
    return point3d.reshape(-1)

#


def get3D(c1, c2, mask1, mask2, glass, _img1=None, _img2=None, drawCentroid=False, drawDimensions=False):

    img1 = copy.deepcopy(_img1)
    img2 = copy.deepcopy(_img2)

    centr1 = getCentroid(c1, mask1)
    centr2 = getCentroid(c2, mask2)
    if centr1 is not None and centr2 is not None:
        glass.centroid = triangulate(c1, c2, centr1, centr2)[:-1].reshape(-1, 3)

        # Draw centroid
        if drawCentroid:

            # Draw 2D centroid of tracking mask
            #cv2.circle(img1, tuple(centr1.astype(int)), 10, (0,128,0), -1)
            #cv2.circle(img2, tuple(centr2.astype(int)), 10, (0,128,0), -1)

            # Draw 3D centroid projected to image
            point1, _ = cv2.projectPoints(glass.centroid, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
            point2, _ = cv2.projectPoints(glass.centroid, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)

            point1 = point1.squeeze().astype(int)
            point2 = point2.squeeze().astype(int)

            cv2.circle(img1, tuple(point1), 6, (128, 0, 0), -1)
            cv2.circle(img2, tuple(point2), 6, (128, 0, 0), -1)

        # Draw height and width lines
        if drawDimensions:

            # Get top/bottom points
            top = copy.deepcopy(glass.centroid)
            bottom = copy.deepcopy(glass.centroid)
            top[0, 2] += glass.h / 2
            bottom[0, 2] -= glass.h / 2
            topC1, _ = cv2.projectPoints(top, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
            bottomC1, _ = cv2.projectPoints(bottom, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
            topC2, _ = cv2.projectPoints(top, c2.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
            bottomC2, _ = cv2.projectPoints(bottom, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
            topC1 = topC1.squeeze().astype(int)
            bottomC1 = bottomC1.squeeze().astype(int)
            topC2 = topC2.squeeze().astype(int)
            bottomC2 = bottomC2.squeeze().astype(int)

            # Get rigth/left points
            right = copy.deepcopy(glass.centroid)
            left = copy.deepcopy(glass.centroid)
            right[0, 0] += glass.w / 2
            left[0, 0] -= glass.w / 2
            rightC1, _ = cv2.projectPoints(right, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
            leftC1, _ = cv2.projectPoints(left, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
            rightC2, _ = cv2.projectPoints(right, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
            leftC2, _ = cv2.projectPoints(left, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
            rightC1 = rightC1.squeeze().astype(int)
            leftC1 = leftC1.squeeze().astype(int)
            rightC2 = rightC2.squeeze().astype(int)
            leftC2 = leftC2.squeeze().astype(int)

            cv2.line(img1, tuple(topC1), tuple(bottomC1), (128, 0, 0), 2)
            cv2.line(img1, tuple(rightC1), tuple(leftC1), (128, 0, 0), 2)
            cv2.line(img2, tuple(topC2), tuple(bottomC2), (128, 0, 0), 2)
            cv2.line(img2, tuple(rightC2), tuple(leftC2), (128, 0, 0), 2)

    return glass, img1, img2

#


def pointsOnContour(p2d_c1, p2d_c2, _contour1, _contour2, _c1, _c2, draw=True):
    contour1 = copy.deepcopy(_contour1)
    contour2 = copy.deepcopy(_contour2)

    c1 = copy.deepcopy(_c1)
    c2 = copy.deepcopy(_c2)

    # Closer re-projected point to contour [ < 5 pixels]
    distances_c1 = cdist(p2d_c1, contour1)
    distances_c2 = cdist(p2d_c2, contour2)

    avg_dist_c1 = np.mean(distances_c1)
    avg_std_c1 = np.std(distances_c1)

    avg_dist_c2 = np.mean(distances_c2)
    avg_std_c2 = np.std(distances_c2)
    print('C1: avgDistance {:.2f} +- {:.2f}'.format(avg_dist_c1, avg_std_c1))
    print('C2: avgDistance {:.2f} +- {:.2f}'.format(avg_dist_c2, avg_std_c2))


def getObjectDimensions(_c1, _c2, roi_list1, roi_list2, centroid, draw=False):

    c1 = copy.deepcopy(_c1)
    c2 = copy.deepcopy(_c2)

    # Radiuses
    step = 0.001  # meters
    minDiameter = 0.005  # meters
    maxDiameter = 0.15  # meters
    radiuses = np.linspace(maxDiameter / 2, minDiameter / 2, num=int((maxDiameter - minDiameter) / step))

    angularStep = 18  # degrees
    angles = np.linspace(0., 359., num=int((359.) / angularStep))

    # Heights
    step = 0.001  # meters
    minHeight = -0.1  # meters
    maxHeight = 0.4  # meters
    estRadius = []
    converged = []

    heights = np.linspace(minHeight, maxHeight, num=int((maxHeight - minHeight) / step))
    # print("step_h:", heights.shape, "step_r:", radiuses.shape)
    for height in heights:
        for rad in radiuses:
            seg1 = copy.deepcopy(c1['seg'])
            seg2 = copy.deepcopy(c2['seg'])

            # Sample 3D circunference
            p3d = []
            for angle_d in angles:
                angle = math.radians(angle_d)
                p3d.append(np.array((centroid[0] + (rad * math.cos(angle)), centroid[1] + (rad * math.sin(angle)), height)).reshape(1, 3))
            p3d = np.array(p3d)

            # Reproject to C1
            p2d_c1, _ = cv2.projectPoints(p3d, c1['extrinsic']['rvec'], c1['extrinsic']['tvec'], c1['intrinsic'], np.array([0., 0., 0., 0., 0.]))
            p2d_c1 = p2d_c1.squeeze().astype(int)

            # Reproject to C2
            p2d_c2, _ = cv2.projectPoints(p3d, c2['extrinsic']['rvec'], c2['extrinsic']['tvec'], c2['intrinsic'], np.array([0., 0., 0., 0., 0.]))
            p2d_c2 = p2d_c2.squeeze().astype(int)

            p2d_c1[:, 1][p2d_c1[:, 1] >= seg1.shape[0]] = seg1.shape[0] - 1
            p2d_c1[:, 0][p2d_c1[:, 0] >= seg1.shape[1]] = seg1.shape[1] - 1
            p2d_c2[:, 1][p2d_c2[:, 1] >= seg2.shape[0]] = seg2.shape[0] - 1
            p2d_c2[:, 0][p2d_c2[:, 0] >= seg2.shape[1]] = seg2.shape[1] - 1
            areIn_c1 = seg1[p2d_c1[:, 1], p2d_c1[:, 0]]
            areIn_c2 = seg2[p2d_c2[:, 1], p2d_c2[:, 0]]
            # areIn_c1 = seg1[p2d_c1[:, 1][p2d_c1[:, 1] < seg1.shape[0]], p2d_c1[:, 0][p2d_c1[:, 0] < seg1.shape[1]]]
            # areIn_c2 = seg2[p2d_c2[:, 1][p2d_c2[:, 1] < seg2.shape[0]], p2d_c2[:, 0][p2d_c2[:, 0] < seg2.shape[1]]]

            assert areIn_c1.shape[0] == areIn_c2.shape[0]
            # if (np.count_nonzero(areIn_c1) == areIn_c1.shape[0]) and (np.count_nonzero(areIn_c2) == areIn_c2.shape[0]):
            # if (np.count_nonzero(areIn_c1) + np.count_nonzero(areIn_c2) >= areIn_c1.shape[0]):
            if (np.count_nonzero(areIn_c1) == areIn_c1.shape[0]) or (np.count_nonzero(areIn_c2) == areIn_c2.shape[0]):
                estRadius.append(rad)
                converged.append(True)
                break
            if rad == minDiameter / 2:
                estRadius.append(rad)
                converged.append(False)
                break

    estRadius = np.array(estRadius)
    converged = np.array(converged)
    estHeights = heights[converged]

    properRadius = estRadius[converged]

    candidate = properRadius
    width_top, width_bottom = -1, -1
    if len(properRadius) > 0:  # task4
        candidate = []
        eps = 1e-3
        r = -1
        for i in range(properRadius.shape[0] - 1):
            dr = properRadius[i + 1] - properRadius[i]
            if abs(dr) < eps and abs(properRadius[i + 1] - r) >= eps:
                candidate.append(properRadius[i])
                r = properRadius[i + 1]

        D = []
        for i in range(len(candidate) - 1):
            dr = candidate[i + 1] - candidate[i]
            D.append((abs(dr), candidate[i + 1]))

        # bottom
        pre, r = D[0]
        for i, (d, r) in enumerate(D):
            if pre > d:
                width_bottom = 2 * r * 1000
                break
            pre = d

        # top
        pre, r = D[-1]
        for i, (d, r) in enumerate(reversed(D)):
            if pre > d:
                width_top = 2 * r * 1000
                break
            pre = d

        print("width:", width_top, width_bottom)

    height = (estHeights[-1] - estHeights[0]) * 1000 if estHeights.shape[0] > 0 else 0
    # capacity = (np.power(average_radius, 2) * height * np.pi)/1000

    capacity = 0
    for i in range(estRadius[converged].shape[0] - 1):
        r1, r2 = estRadius[converged][i], estRadius[converged][i + 1]
        h1, h2 = heights[converged][i], heights[converged][i + 1]
        capacity += np.pi * abs(h2 - h1) / 3 * (np.power(r1, 2) + np.power(r2, 2) + r1 * r2)

    capacity *= 1000**2
    print("capacity:", capacity)

    img1 = copy.deepcopy(c1['rgb'])
    img2 = copy.deepcopy(c2['rgb'])

    masked1 = np.zeros_like(img1)
    masked2 = np.zeros_like(img2)

    for i in range(3):
        masked1[:, :, i] = img1[:, :, i] * c1['seg']
        masked2[:, :, i] = img2[:, :, i] * c2['seg']

    img1_with_bbox = draw_bbox(roi_extract(roi_list1), img1)
    img2_with_bbox = draw_bbox(roi_extract(roi_list2), img2)

    vis = np.concatenate((img1, img2), axis=1)

    # Draw final dimensions
    if draw:

        for i, rad in enumerate(estRadius):

            p3d = []
            for angle_d in angles:
                angle = math.radians(angle_d)
                p3d.append(np.array((centroid[0] + (rad * math.cos(angle)), centroid[1] + (rad * math.sin(angle)), heights[i])).reshape(1, 3))
            p3d = np.array(p3d)

            # Reproject to C1
            p2d_c1, _ = cv2.projectPoints(p3d, c1['extrinsic']['rvec'], c1['extrinsic']['tvec'], c1['intrinsic'], np.array([0., 0., 0., 0., 0.]))
            p2d_c1 = p2d_c1.squeeze().astype(int)

            # Reproject to C2
            p2d_c2, _ = cv2.projectPoints(p3d, c2['extrinsic']['rvec'], c2['extrinsic']['tvec'], c2['intrinsic'], np.array([0., 0., 0., 0., 0.]))
            p2d_c2 = p2d_c2.squeeze().astype(int)

            # Check if imaged points are in the segmentation
            p2d_c1[:, 1][p2d_c1[:, 1] >= seg1.shape[0]] = seg1.shape[0] - 1
            p2d_c1[:, 0][p2d_c1[:, 0] >= seg1.shape[1]] = seg1.shape[1] - 1
            p2d_c2[:, 1][p2d_c2[:, 1] >= seg2.shape[0]] = seg2.shape[0] - 1
            p2d_c2[:, 0][p2d_c2[:, 0] >= seg2.shape[1]] = seg2.shape[1] - 1
            areIn_c1 = seg1[p2d_c1[:, 1], p2d_c1[:, 0]]
            areIn_c2 = seg2[p2d_c2[:, 1], p2d_c2[:, 0]]

            for p, isIn in zip(p2d_c1, areIn_c1):
                if isIn:
                    cv2.circle(img1, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                # else:
                # 	cv2.circle(img1, (int(p[0]), int(p[1])), 2, (0,0,255), -1)

            for p, isIn in zip(p2d_c2, areIn_c2):
                if isIn:
                    cv2.circle(img2, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                # else:
                # 	cv2.circle(img2, (int(p[0]), int(p[1])), 2, (0,0,255), -1)

        # mask画像と左右の画像を縦方向にconcat
        img = np.concatenate((img1, img2), axis=1)
        masked = np.concatenate((masked1, masked2), axis=1)
        img_with_bbox = np.concatenate((img1_with_bbox, img2_with_bbox), axis=1)
        vis = np.concatenate((img_with_bbox, masked, img))

    return height, (width_top, width_bottom), vis, capacity, candidate
