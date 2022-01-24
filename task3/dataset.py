import random
import json
import os
import re


class Dataset:
    def __init__(self, data_path):
        self.annotations = None
        self.dict, self.set = None, None
        self.data_path = data_path

    def get_calib_path(self, fid=None, view=None):
        calibration_path = os.path.join(self.data_path, 'view{}'.format(view), 'calib')
        path = calibration_path + '/{}.pickle'.format(fid, view)
        assert os.path.exists(path), "Can't find path " + path
        return path

    def get_video_path(self, fid=None, view=None):
        rgb_path = os.path.join(self.data_path, 'view{}'.format(view), 'rgb')
        path = "{}/{}.mp4".format(rgb_path, fid, view)
        assert os.path.exists(path), "Can't find path " + path
        return path

    def get_all_fileids(self):
        calibration_path = os.path.join(self.data_path, 'view1', 'calib')
        pattern = re.compile(r"([\w\d_]+).pickle")
        file_id_list = [pattern.match(f).group(1) for f in os.listdir(calibration_path) if pattern.match(f)]
        return file_id_list


class TestDataset(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)


class TrainDataset(Dataset):
    def __init__(self, data_path, ratio=0.8):
        super().__init__(data_path)
        self.ratio = ratio
        self.annotations = self.load_annotations()
        self.dict, self.set = self.generate_dataset()

    def load_annotations(self):
        with open("{}/ccm_train_annotation.json".format(self.data_path)) as f:
            df = json.load(f)
            df = df["annotations"]
        return df

    def generate_dataset(self):
        obj_to_idxs = {}
        data_count = 0
        for obj in self.annotations:
            id, cid = int(obj["id"]), int(obj["container id"])
            if cid not in obj_to_idxs.keys(): obj_to_idxs[cid] = []
            obj_to_idxs[cid].append(id)
            data_count += 1

        random.seed(0)
        train_set = []
        train_dict = {}
        for cid, idxs in obj_to_idxs.items():
            idxs.sort()
            train = random.sample(idxs, len(idxs))[:int(len(idxs) * self.ratio)]
            train_dict[cid] = train
            train_set.extend(train)
        return train_dict, set(train_set)

    def get_container_mass_data(self):
        X = list(self.set)
        _y = []
        for i in range(len(self.annotations)):
            row = self.annotations[i]
            if row["id"] in self.set:
                _y.append((row["id"], row["container mass"]))

        _y.sort()
        y = [t[1] for t in _y]
        return (X, y)


class ValidationDataset(Dataset):
    def __init__(self, data_path, ratio=0.2):
        super().__init__(data_path)
        self.ratio = ratio
        self.annotations = self.load_annotations()
        self.dict, self.set = self.generate_dataset()

    def load_annotations(self):
        with open("{}/ccm_train_annotation.json".format(self.data_path)) as f:
            df = json.load(f)
            df = df["annotations"]
        return df

    def generate_dataset(self):
        obj_to_idxs = {}
        data_count = 0
        for obj in self.annotations:
            id, cid = int(obj["id"]), int(obj["container id"])
            if cid not in obj_to_idxs.keys(): obj_to_idxs[cid] = []
            obj_to_idxs[cid].append(id)
            data_count += 1

        random.seed(0)
        val_set = []
        val_dict = {}
        for cid, idxs in obj_to_idxs.items():
            idxs.sort()
            val = random.sample(idxs, len(idxs))[int(len(idxs) * (1 - self.ratio)):]
            val_dict[cid] = val
            val_set.extend(val)
        return val_dict, set(val_set)

    def get_container_mass_data(self):
        X = list(self.set)
        _y = []
        for i in range(len(self.annotations)):
            row = self.annotations[i]
            if row["id"] in self.set:
                _y.append((row["id"], row["container mass"]))

        _y.sort()
        y = [t[0] for t in _y]
        return (X, y)


class DebugDataset(ValidationDataset):
    def __init__(self, data_path, ratio=0.2):
        super().__init__(data_path)
        self.ratio = ratio
        self.annotations = self.load_annotations()
        self.dict, self.set = self.generate_validation_dataset()

    def generate_validation_dataset(self):
        obj_to_idxs = {}
        data_count = 0
        for obj in self.annotations:
            id, cid = int(obj["id"]), int(obj["container id"])
            if cid not in obj_to_idxs.keys(): obj_to_idxs[cid] = []
            obj_to_idxs[cid].append(id)
            data_count += 1

        random.seed(0)
        val_set = []
        val_dict = {i: [] for i in range(20)}
        for cid, idxs in obj_to_idxs.items():
            if cid != 2: continue
            idxs.sort()
            val = random.sample(idxs, int(len(idxs) * self.ratio))
            val_dict[cid] = val
            val_set.extend(val)
        return val_dict, set(val_set)
