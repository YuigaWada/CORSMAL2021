import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from corsmal_challenge.data.audio import load_wav
from corsmal_challenge.utils import fix_random_seeds


class AudioDataset(torch.utils.data.Dataset):
    """audio dataset"""

    def __init__(
        self,
        path_to_data: Path = Path("./data/train/"),
        path_to_annotation_file: Path = Path("./data/train/ccm_train_annotation.json"),
        seed: int = 0,
        valset_rate: int = 2,
        mv_val2train: int = 0,
        train: bool = True,
        query: str = "type",  # or "level"
        random_crop: bool = False,
        strong_crop: bool = False,
        clip_end: bool = False,
    ):
        """constructor of AudioDataset

        Args:
            path_to_data (Path, optional): Path to data directory. Defaults to Path("./data/train/").
            path_to_annotation_file (Path, optional): Path to annotation file directory.\\
                Defaults to Path("./data/train/ccm_train_annotation.json").
            seed (int, optional): Random seed to fix randomness. Defaults to 0.
            valset_rate (int, optional): Rate of validation set (valset_rate / 10). Defaults to 2.
            mv_val2train (int, optional): If you set, move subset of validation set to training set. Defaults to 0.
            train (bool, optional): If you want this instance to behave as training dataset, enable it.\\
                Defaults to True.
            query (str, optional): Specifying query type, choose "type" or "level". Defaults to "type".
            strong_crop (bool, optional): When enabled, do strong data augmentation. Defaults to False.
            clip_end (bool, optional): When enabled, do not crop the tail of sequence on data augmentation.\\
                Defaults to False.
        """
        fix_random_seeds(seed)

        self.seed = seed
        self.valset_rate: int = valset_rate
        self.mv_val2train: int = mv_val2train

        self.type_label = {0: "none", 1: "pasta", 2: "rice", 3: "water"}
        self.level_label = {0: "empty", 1: "half-full", 2: "full"}
        self.annotations = self._get_annotations(path_to_annotation_file)
        self.val_idx, self.train_idx = self._divide_idx()
        self.audio_path: Path = path_to_data / "audio"
        self.train: bool = train
        self.query: str = query
        self.random_crop: bool = random_crop
        self.strong_crop: bool = strong_crop
        self.clip_end: bool = clip_end

    def _get_annotations(self, path_to_annotation_file) -> List[Dict[str, int]]:
        """Invoked by constructor to get summarized annotation dictionary"""
        with open(str(path_to_annotation_file), "r") as f:
            dic: Dict = json.load(f)
        annotations = [{"type": data["filling type"], "level": data["filling level"]} for data in dic["annotations"]]
        return annotations

    def _get_classified_data(self) -> List[List[List[int]]]:
        """Invoked by self._divide_idx() to get classified indexes"""
        classified_data: List[List[List[int]]] = [[[] for j in range(3)] for i in range(4)]
        for idx, data in enumerate(self.annotations):
            filling_type = data["type"]
            filling_level = data["level"]
            classified_data[filling_type][filling_level].append(idx)
        return classified_data

    def _divide_idx(self) -> Tuple[List, List]:
        """Invoked by constructor to get index list of validation set & training set"""
        classified_data: List[List[List[int]]] = self._get_classified_data()
        self.annotations
        val_idx: List[int] = []
        train_idx: List[int] = []
        for d in classified_data:
            for lis in d:
                val_list = set(random.sample(lis, len(lis) // 10 * self.valset_rate))
                for id in lis:
                    if id == 377:  # anomaly
                        continue
                    if id in val_list:
                        val_idx.append(id)
                    else:
                        train_idx.append(id)
        for i in range(self.mv_val2train):
            train_idx.append(val_idx.pop())
        return val_idx, train_idx

    def __len__(self):
        """
        Get length of dataset. If you enable self.train, return the number of training dataset\\
        and if you disable self.train, return the number of validation dataset.
        """
        return len(self.train_idx) if self.train else len(self.val_idx)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """Get data from specified index

        Args:
            idx ([type]): Specified index

        Returns:
            Tuple[torch.Tensor, int]: Tensor (channels, sequence, embedding dim.), label
        """
        id: int = self.train_idx[idx] if self.train else self.val_idx[idx]
        data_path: Path = self.audio_path / (str(id).zfill(6) + ".wav")
        label = self.annotations[id][self.query]
        spectrogram = load_wav(data_path).generate_mel_spectrogram().log2()

        if self.random_crop:
            sequence_len: int = spectrogram.shape[-1]
            if self.strong_crop:
                start = random.randrange(0, sequence_len // 10 * 4)
                end = random.randrange(sequence_len // 10 * 6, sequence_len)
            else:
                start = random.randrange(0, sequence_len // 10 * 2)
                end = random.randrange(sequence_len // 10 * 8, sequence_len)
            if self.clip_end:
                start += sequence_len - end
                end = sequence_len - 1
            return spectrogram[:, :, start : end + 1].transpose(-1, -2), label

        return spectrogram.transpose(-1, -2), label
