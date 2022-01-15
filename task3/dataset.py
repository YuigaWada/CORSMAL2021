import random
import json


class Dataset:
    def __init__(self, data_path):
        self.annotations = None
        self.dict, self.set = None, None
        self.data_path = data_path


class TrainDataset(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_annotations(self):
        with open("{}/ccm_train_annotation.json".format(self.data_path)) as f:
            df = json.load(f)
            df = df["annotations"]
        return df


class ValidationDataset(TrainDataset):
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
        val_dict = {}
        for cid, idxs in obj_to_idxs.items():
            idxs.sort()
            val = random.sample(idxs, len(idxs))[int(len(idxs) * (1 - self.ratio)):]  # 昨年のHVRLのコード参照
            val_dict[cid] = val
            val_set.extend(val)
        return val_dict, set(val_set)


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
