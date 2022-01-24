import os
import time
import cv2
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from task3.video_processing import DynamicVideoProcessing


def get_crop_region(mask):
    res = [1 << 30] * 4
    for j in range(4):
        for i in range(mask.shape[1]):
            l, r = 0, mask.shape[0]
            while r - l > 1:
                mid = (l + r) // 2
                if mask[:mid, i].sum() == 0:
                    l = mid
                else:
                    r = mid
            if r != mask.shape[0]:
                res[j] = min(res[j], l)

        mask = np.rot90(mask)
        if res[j] == 1 << 30:
            res[j] = 0

    return res


def get_boundary(mask, i):
    l, r = 0, mask.shape[1]
    while r - l > 1:
        mid = (l + r) // 2
        if mask[i, :mid].sum() == 0:
            l = mid
        else:
            r = mid

    return l if r != mask.shape[1] else None


def get_line_section(mask, i):
    l, r = -1, mask.shape[1] - 1
    x = get_boundary(mask, i)
    if x is None: return (None, None)
    l = x

    _mask = np.rot90(np.rot90(mask))
    x = get_boundary(_mask, mask.shape[0] - 1 - i)
    if x is not None:
        r = _mask.shape[1] - x

    return l, r


def preprocessing(mask):  # 回転軸を決定して手の凹みを補間. 完全に手で遮られているところは前の情報を使う(TODO)
    t = 0
    l, r = -1, mask.shape[1] - 1
    for i in range(mask.shape[0]):
        l, r = get_line_section(mask, i)
        if l is not None:
            t = i
            break

    if l == -1: return mask

    size = r - l + 1
    axis = l + size // 2  # rotation-axis

    pre = (l, r)
    empty = []
    for i in range(t + 1, mask.shape[0]):  # mask[:axis], mask[axis:]
        l, r = get_line_section(mask, i)

        if l is not None:
            left = axis - l + 1
            right = r - axis + 1

            if l <= axis <= r:
                dx = abs(right - left)
                if left <= right:
                    l = max(0, l - dx)
                else:
                    r = min(mask.shape[1] - 1, r + dx)

            elif axis <= l <= r:
                l = max(0, axis - right)
            else:
                r = min(mask.shape[1] - 1, axis + left)

            mask[i, l:r + 1] = 255
        else:
            empty.append((i, pre))

        pre = (l, r)

    return mask
    # todo: emptyの処理


class MaskDataset(Dataset):
    def __init__(self, dataset, df, is_train=True, is_test=False):
        use_annotation = False
        video_processing = DynamicVideoProcessing(output_path='outputs', dataset=dataset)
        detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        detectionModel.eval()
        detectionModel.cuda()
        x, y = dataset.get_container_mass_data()

        self.images = []
        self.dimension_vector_list = []
        self.Y = []
        self.file_ids = []
        self.time_table = {}
        path = './cropped' if is_train else "./cropped_val"
        if not os.path.exists(path): os.makedirs(path)
        for i, fid in enumerate(x):
            print("[task4.MaskDataset] video-processing: {}/{}".format(i, len(x)))
            fid_str = str(1000000 + fid)[1:]

            start_time = time.process_time()
            files = [f for f in os.listdir(path) if f == 'seg{}.pt'.format(fid_str)]
            row = df[df["Configuration ID"] == fid]
            if len(row["Configuration ID"]) == 0: continue

            if use_annotation:
                pass
                # d = validation.annotations[fid]
                # width_top = d["width at the top"]
                # width_top = d["width at the bottom"]
                # height = d["height"]
            else:
                width_top = list(row["Width at the top"])[0]
                width_bottom = list(row["Width at the bottom"])[0]
                height = list(row["Height"])[0]

            if len(files) > 0:
                resized = torch.load('{}/seg{}.pt'.format(path, fid_str,))
                self.images.append(resized)
                self.dimension_vector_list.append((width_top, width_bottom, height))
                self.file_ids.append(fid)
                self.Y.append(y[i])
                elapsed_time = time.process_time() - start_time
                self.time_table[fid] = elapsed_time * 1000
                continue

            view = 3
            c, roi = video_processing.prepare_data(detectionModel, fid_str, [view], "task4")
            img = c[view]['rgb']
            seg = c[view]["seg"]
            if seg is None: 
                if is_test:
                    self.images.append(None)
                    self.dimension_vector_list.append(None)
                    self.file_ids.append(fid)
                    self.Y.append(y[i])
                continue

            seg = seg / 255
            y1, x2, y2, x1 = get_crop_region(seg)
            y2 = seg.shape[0] - y2
            x2 = seg.shape[1] - x2
            h, w = y2 - y1, x2 - x1
            # print("region", y1, x1, y2, x2)

            for k in range(3):
                img[:, :, k] = img[:, :, k] * seg

            img = seg * 255
            pad = 15
            y1, y2 = max(0, y1 - pad), min(img.shape[0], y2 + pad)
            x1, x2 = max(0, x1 - pad), min(img.shape[1], x2 + pad)
            cropped = img[y1:y2, x1:x2]
            cropped = preprocessing(cropped)
            cv2.imwrite('{}/seg{}_{}_{}_.jpeg'.format(path, fid_str, h, w), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            resized = cv2.resize(cropped / 255, (112, 112))
            resized = torch.Tensor(resized).unsqueeze(0)

            self.images.append(resized)
            self.dimension_vector_list.append((width_top, width_bottom, height))
            self.file_ids.append(fid)
            self.Y.append(y[i])

            cv2.imwrite('{}/bbox{}_{}_{}_.jpeg'.format(path, fid_str, h, w), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            torch.save(resized, '{}/seg{}.pt'.format(path, fid_str,))
            # _resized = torch.load('{}/seg{}.pt'.format(path, fid_str,))
            # assert torch.equal(resized,_resized)

            # measure time
            elapsed_time = time.process_time() - start_time
            self.time_table[fid] = elapsed_time * 1000

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.images[idx] is None: return (None, None), None, torch.Tensor([self.file_ids[idx]])
        image = torch.Tensor(self.images[idx])
        dimension_vector = torch.Tensor(self.dimension_vector_list[idx])
        Y = torch.Tensor([self.Y[idx]])
        file_id = torch.Tensor([self.file_ids[idx]])

        return (image, dimension_vector), Y, file_id

    def get_elapsed_time(self, file_id): # [msec]
        if file_id not in self.time_table: return 0
        return self.time_table[file_id]
