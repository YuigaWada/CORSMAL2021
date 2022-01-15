##################################################################################
#        Author: Ricardo Sanchez Matilla
#         Email: ricardo.sanchezmatilla@qmul.ac.uk
#  Created Date: 2020/02/13
# Modified Date: 2020/02/28

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
import numpy as np
import copy
import cv2

import torch
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trf = T.Compose([T.ToPILImage(), T.ToTensor()])


def imageSegmentation(detectionModel, _img):
    labels_to_avoid = [1, 15, 62, 63, 67, 68, 69, 77]
    img = copy.deepcopy(_img)
    img = img[:, :, [2, 1, 0]]

    candidate = []
    output = detectionModel([trf(img).to(device)])[0]
    segs = {}
    for i in range(0, len(output['labels'])):
        if not is_clear(_img, output['boxes'][i]): continue
        if output['scores'][i] >= 0.5 and output['labels'][i] in [44, 46, 47, 84, 86]:  # bottle, wine glass, cup, book, vas
            if len(candidate) >= 3: break
            l = -1
            seg = np.uint8(255. * output['masks'][i, :, :, :].detach().cpu().numpy().squeeze())
            seg = ((seg >= 128) * 255).astype(np.uint8)
            x = np.sum(seg, axis=0)
            for j in range(x.shape[0]):
                if x[j]: l = max(l, j)

            candidate.append((l, i))
            segs[i] = seg

    candidate.sort()
    roi = []
    M = -1
    weights = [0.5, 1, 1.5]
    for k, (l, i) in enumerate(candidate):
        output['scores'][i] = min(1, output['scores'][i] * weights[k])
        M = max(M, output['scores'][i])

    for k, (l, i) in enumerate(candidate):
        if output['scores'][i] == M:
            return segs[i], output

    return None, output


def is_clear(frame, bbox):
    height, width = frame.shape[0], frame.shape[1]
    center_x = (bbox[0] + bbox[2]) / 2
    return width * 1 / 4 <= center_x <= width * 3 / 4
