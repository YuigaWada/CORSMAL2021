import cv2
import shutil
import os
import copy
import pandas as pd
import numpy as np

# Mask-RCNN
coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def draw_bbox(roi_list, image):
    image = copy.deepcopy(image)
    for roi in roi_list:
        score = roi[1]
        if score < 0.3: continue
        box = roi[2]
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        label = roi[0]
        display_txt = "%s: %.1f%%" % (coco_names[label], 100 * score)
        tl = 3
        color = (0, 0, 255)
        cv2.rectangle(image, c1, c2, color, thickness=tl)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1)  # filled
        cv2.putText(image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return image


def roi_extract(output):
    roi_list = []
    width = 1280
    # high_score_container=True
    for i in range(0, len(output['labels'])):
        if output['boxes'][i][2] <= width / 4 or output['boxes'][i][0] >= width / 4 * 3:
            continue
        if output['labels'][i] in [44, 46, 47, 86]:
            seg = np.uint8(255. * output['masks'][i, :, :, :].detach().cpu().numpy().squeeze())
            seg = ((seg >= 128) * 255).astype(np.uint8)
            roi_list.append([output['labels'][i].item(), output['scores'][i].item(), output['boxes'][i], seg])
    return roi_list


# Scores
def calc_final_estimation(dataset, average_training_set, phase, tags):
    path_to_load = 'results/'
    combined_file_path = f'estimation_combination_{phase}.csv'

    dfs = [pd.read_csv(path_to_load + f'estimation_{tag}_{phase}.csv') for tag in tags]
    combined_file = pd.DataFrame(data=dfs[0][['fileName', 'capacity[mL]', 'score']].values, columns=['fileName', 'capacity[mL]', 'score'])

    scores, cscores = [], {}
    for index, row in dfs[0].iterrows():
        candidate = [dfs[i]['capacity[mL]'][index] for i in range(len(dfs))]
        candidate = [x for x in candidate if x != average_training_set]
        combined_file['capacity[mL]'][index] = np.mean(candidate) if len(candidate) > 0 else average_training_set

        cid = int(row["fileName"][2:].split("_")[0])
        fileid = int(row["fileName"][2:].split("_")[1][:-4])
        answer = float(dataset.annotations[fileid]["container capacity"])

        capacity = combined_file["capacity[mL]"][index]
        score = np.exp(-np.abs(capacity - answer) / answer)
        combined_file["score"][index] = score
        if cid not in cscores.keys(): cscores[cid] = []
        cscores[cid].append(score)
        scores.append(score)

    combined_file.to_csv(path_to_load + combined_file_path)
    return scores, cscores
