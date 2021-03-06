"""path resolving"""
import sys
from pathlib import Path

current_dir: Path = Path().cwd().resolve()
project_root: Path = current_dir.parent
data_dir: Path = project_root / "data" / "train"
audio_dir: Path = data_dir / "audio"

sys.path.append(str(project_root))


"""main"""
import json  # noqa (E402)
import random  # noqa (E402)
from typing import Dict, List, Tuple  # noqa (E402)

import matplotlib.pyplot as plt  # noqa (E402)
import numpy as np  # noqa (E402)
import torch  # noqa (E402)
from torch import nn  # noqa (E402)

from corsmal_challenge.data.data_loader import (  # noqa (E402)
    ReproducibleDataLoader as DataLoader,
)
from corsmal_challenge.data.dataset import Labels, SimpleAudioDataset  # noqa (E402)
from corsmal_challenge.models.task1_2 import TaskChallenger3  # noqa (E402)
from corsmal_challenge.train.train_val import classification_loop  # noqa (E402)
from corsmal_challenge.utils import fix_random_seeds  # noqa (E402)

RAND_SEED = 0
FOLDS = 9
# FOLD_SKIP_FREQ = 9
FOLD_SKIP_FREQ = 3
EPOCH = 100
# EPOCH = 120


if __name__ == "__main__":
    fix_random_seeds(RAND_SEED)

    # divide dataset
    ids: List[List[int]] = [[] for _ in range(FOLDS)]
    labels: List[List[Labels]] = [[] for _ in range(FOLDS)]
    type_label = {0: "no content", 1: "pasta", 2: "rice", 3: "water"}
    level_label = {0: "0%", 1: "50%", 2: "90%"}

    with open(str(data_dir / "ccm_train_annotation.json"), "r") as f:
        dic = json.load(f)
        annotations: Dict = dic["annotations"]

    classified_data: List[List[List[int]]] = [[[] for j in range(3)] for i in range(4)]
    for data in annotations:
        id = data["id"]
        filling_type = data["filling type"]
        filling_level = data["filling level"]
        classified_data[filling_type][filling_level].append(id)

    ids_labels: List[List[Tuple[int, Labels]]] = [[] for _ in range(FOLDS)]
    for typ, d in enumerate(classified_data):  # type
        for lvl, id_list in enumerate(d):  # level
            fld = 0
            for idx in id_list:
                ids_labels[fld].append((idx, Labels(typ, lvl)))
                fld = (fld + 1) % FOLDS
    for fld in range(FOLDS):
        random.shuffle(ids_labels[fld])
        for id, label in ids_labels[fld]:
            ids[fld].append(id)
            labels[fld].append(label)

    # cross validation
    loss_acc_summaries: List[str] = []
    for fold_id in range(FOLDS):
        if fold_id % FOLD_SKIP_FREQ != 0:
            print("")
            print(f"FOLD.{fold_id + 1} => SKIP")
            continue
        else:
            print("")
            print(f"FOLD.{fold_id + 1} => START")

        train_idx: List[int] = []
        train_label: List[Labels] = []
        for i in list(range(FOLDS))[:fold_id] + list(range(FOLDS))[fold_id + 1 :]:
            train_idx.extend(ids[i])
            train_label.extend(labels[i])
        val_idx: List[int] = ids[fold_id]
        val_label: List[Labels] = labels[fold_id]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaskChallenger3()
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.5, lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.80)
        train_dataset = SimpleAudioDataset(train_idx, train_label, data_dir)
        val_dataset = SimpleAudioDataset(val_idx, val_label, data_dir)
        train_dataloader = DataLoader(train_dataset, specified_seed=RAND_SEED, shuffle=True)
        val_dataloader = DataLoader(val_dataset, specified_seed=RAND_SEED, shuffle=True)

        train_loss_t1: List[float] = []
        val_loss_t1: List[float] = []
        train_loss_t2: List[float] = []
        val_loss_t2: List[float] = []

        min_val_loss_sum_t1_t2: float = float("inf")
        best_epoch_id = -1
        best_loss_pair = (12.34, 45.67)
        best_acc_pair = (0.0, 0.0)

        plt.figure()
        for step in range(EPOCH):
            train_dataset.query = "type"
            val_dataset.query = "type"
            train_dataset.random_crop = True
            train_dataset.strong_crop = True
            train_dataset.clip_end = False
            model.task_id = 2

            tup = classification_loop(  # type: ignore
                device,
                model,
                loss_fn,
                optimizer,
                train_dataloader,
                val_dataloader,
                enable_amp=True,
            )
            metrics = tup[1]
            train_loss_t2.append(metrics["train loss"])
            val_loss_t2.append(metrics["val loss"])
            val_acc_t2 = metrics["val accuracy"]
            print(f"  EPOCH.{str(step).zfill(3)}  <task2> Type Classification")
            print("               ", end="")
            for mtrc, value in metrics.items():
                print(f"{mtrc}: {value:.3f}", end=" | ")
            print("")

            train_dataset.query = "level"
            val_dataset.query = "level"
            train_dataset.random_crop = False
            train_dataset.strong_crop = False
            train_dataset.clip_end = True
            model.task_id = 1

            tup = classification_loop(  # type: ignore
                device,
                model,
                loss_fn,
                optimizer,
                train_dataloader,
                val_dataloader,
                enable_amp=True,
            )
            metrics = tup[1]
            train_loss_t1.append(metrics["train loss"])
            val_loss_t1.append(metrics["val loss"])
            val_acc_t1 = metrics["val accuracy"]
            print("             <task1> Level Classification")
            print("               ", end="")
            for mtrc, value in metrics.items():
                print(f"{mtrc}: {value:.3f}", end=" | ")
            print("")

            lr_scheduler.step()

            if min_val_loss_sum_t1_t2 > max(0.15, val_loss_t1[-1]) + max(0.15, val_loss_t2[-1]):
                min_val_loss_sum_t1_t2 = val_loss_t1[-1] + val_loss_t2[-1]
                best_loss_pair = val_loss_t1[-1], val_loss_t2[-1]
                best_acc_pair = val_acc_t1, val_acc_t2
                best_epoch_id = step  # 0-indexed
                torch.save(model.state_dict(), current_dir / f"20220120-cv-result-fold{fold_id + 1}.pt")

        plt.plot(train_loss_t1, label="train loss: t1")
        plt.plot(val_loss_t1, label="val loss: t1")
        plt.plot(train_loss_t2, label="train loss: t2")
        plt.plot(val_loss_t2, label="val loss: t2")
        plt.legend()
        plt.savefig(str(current_dir / f"20220120-cv-result-fold{fold_id + 1}.png"))

        loss_acc_summaries.append(f"Fold.{fold_id + 1}")
        loss_acc_summaries.append(f"  on epoch.{best_epoch_id}")
        loss_acc_summaries.append(f"  best (val_loss_t1, val_loss_t2) pair is {best_loss_pair}!")
        loss_acc_summaries.append(f"  then (val_acc_t1, val_acc_t2) pair is {best_acc_pair}!")
        print(f"FOLD.{fold_id + 1} => FIN.")

    for summary in loss_acc_summaries:
        print(summary)

# STDOUT
# ```
# :
# Fold.1
#   on epoch.47
#   best (val_loss_t1, val_loss_t2) pair is (0.45398750652116415, 0.0869900718096771)!
#   then (val_acc_t1, val_acc_t2) pair is (0.75, 0.9736842105263158)!
# Fold.4
#   on epoch.10
#   best (val_loss_t1, val_loss_t2) pair is (0.7726110924329412, 0.2768814844258534)!
#   then (val_acc_t1, val_acc_t2) pair is (0.6710526315789473, 0.868421052631579)!
# Fold.7
#   on epoch.24
#   best (val_loss_t1, val_loss_t2) pair is (0.5331498896304862, 0.18764697292231416)!
#   then (val_acc_t1, val_acc_t2) pair is (0.6578947368421053, 0.9210526315789473)!
# (miniconda3-4.7.12) 
# ```
