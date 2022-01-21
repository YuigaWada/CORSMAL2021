import glob
import sys
import argparse
import os


from abc import ABCMeta, abstractmethod
from libs._3d.projection import *
from libs.detection.detection import imageSegmentation
from config import *


class AbstractVideoProcessing:
    @abstractmethod
    def prepare_data(self, detectionModel, rgb_path, fid, views, tag):
        pass


class DynamicVideoProcessing(AbstractVideoProcessing):
    def __init__(self, args, output_path):
        self.c = [dict.fromkeys(['rgb', 'seg', 'intrinsic', 'extrinsic']) for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.roi = [[] for _ in range(VIEW_COUNT + 1)]  # camera1-3
        self.args = args
        self.output_path = output_path

    def prepare_data(self, detectionModel, args, fid, views, tag):
        rgb_path = os.path.join(self.args.path2data, self.args.object, 'rgb')
        caps = self.read_caps(rgb_path, fid, views)
        frame_idx = self.get_best_frame(detectionModel, caps)
        print("frame_idx:", frame_idx)
        for i, cap in enumerate(caps):
            view = views[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, f = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = cap.read()
            self.c[view]['rgb'] = f
            self.c[view]['seg'], self.roi[view] = imageSegmentation(detectionModel, self.c[view]['rgb'])

            img = copy.deepcopy(f)
            img = draw_bbox(roi_extract(self.roi[view]), img)
            cv2.imwrite('{}/bbox/id{}_{}_c{}_{}.jpeg'.format(self.output_path, self.args.object, fid, view, tag), img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        return self.c, self.roi

    def get_best_frame(self, detectionModel, caps):
        fps, video_frame = 1 << 30, 1 << 30
        for _ in range(len(caps)):
            fps = min(fps, caps[0].get(cv2.CAP_PROP_FPS))
            video_frame = min(video_frame, caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        candidate = []
        for i in range(int(video_frame)):
            valid = True
            frames = []
            for j in range(len(caps)):
                ret, frame = caps[j].read()
                valid &= ret
                frames.append(frame)

            if not valid: break
            if i % fps == 0 or i == video_frame - 1:  # 1sごと
                is_clear = True
                self.current = i
                mseg = [[] for _ in range(len(caps))]
                mroi = [[] for _ in range(len(caps))]
                for j in range(len(caps)):
                    mseg[j], mroi[j] = imageSegmentation(detectionModel, frames[j])
                    is_clear &= mseg[j] is not None and self.has_clear_roi(frames[j], mroi[j])

                if is_clear:
                    score = self.calc_all_detect_score(mseg, mroi)
                    print(i, frame.shape, score)
                    candidate.append((i, score))

        if len(candidate) == 0:
            return 0

        candidate.sort(key=lambda x: x[1])
        return candidate[-1][0]

    def has_clear_roi(self, frame, output):
        candidate = []
        height, width = frame.shape[0], frame.shape[1]
        for i in range(0, len(output['labels'])):
            if output["scores"][i] < 0.3: continue
            center_x = (output['boxes'][i][0] + output['boxes'][i][2]) / 2
            center_y = (output['boxes'][i][1] + output['boxes'][i][3]) / 2
            candidate.append((output["scores"][i], center_x, center_y))

        candidate.sort(reverse=True)
        for _, center_x, center_y in candidate[0:min(len(candidate), 3)]:
            if width * 1 / 3 <= center_x <= width * 2 / 3 and height * 1 / 3 <= center_y <= height * 2 / 3:
                return True

        return False

    def assm_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        assert (h > 0) and (w > 0)
        return y1, x1, h, w

    def calc_all_detect_score(self, mseg, outputs):
        scores = 0
        for i in range(len(mseg)):
            score = self.calc_detect_score(mseg[i], outputs[i])
            scores += score
        return scores

    def calc_detect_score(self, seg, output):
        bboxes = []
        for i in range(0, len(output['labels'])):
            if output["scores"][i] < 0.3: continue
            bboxes.append((output["scores"][i], output['boxes'][i]))

        bboxes.sort(reverse=True)
        tops = bboxes[0:min(3, len(bboxes))]

        score = np.sum(seg / 255)
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                _, bbox1 = tops[i]
                _, bbox2 = tops[j]
                if bbox1[0] > bbox2[0]:  # x
                    bbox1, bbox2 = bbox2, bbox1  # swap

                y1, x1, h1, w1 = self.assm_bbox(bbox1.cpu().detach().numpy())
                y2, x2, h2, w2 = self.assm_bbox(bbox2.cpu().detach().numpy())
                if y1 < y2:
                    ds = max(0, x2 + w2 - x1) * max(0, y1 + h1 - y2)
                else:
                    ds = max(0, x2 + w2 - x1) * max(0, y2 + h2 - y1)

                score -= ds  # penalty

        # return max(0,score)
        return score

    def read_caps(self, rgb_path, fid, views):
        caps = []
        for view in views:
            path = "{}/{}_c{}.mp4".format(rgb_path, fid, view)
            assert os.path.exists(path)
            cap = cv2.VideoCapture(path)
            caps.append(cap)
        return caps
