"""path resolving"""
import sys
from pathlib import Path

current_dir: Path = Path().cwd().resolve()
project_root: Path = current_dir.parent
data_dir: Path = project_root / "data" / "train"

sys.path.append(str(project_root))


"""main"""
from typing import Dict, List, NamedTuple  # noqa (E402)
import time  # noqa (E402)
import matplotlib.pyplot as plt  # noqa (E402)
import json  # noqa (E402)
import torch  # noqa (E402)
from copy import deepcopy  # noqa (E402)
from torch import nn  # noqa (E402)

from corsmal_challenge.data.data_loader import (  # noqa (E402)
    ReproducibleDataLoader as DataLoader,
)
from corsmal_challenge.data.dataset import AudioDataset  # noqa (E402)
from corsmal_challenge.models.task1_2 import TaskChallenger3  # noqa (E402)
from corsmal_challenge.train.train_val import classification_loop  # noqa (E402)
from corsmal_challenge.utils import fix_random_seeds  # noqa (E402)

RAND_SEED = 0
EPOCH = 100


class Estimation:
    def __init__(self, valid, total):
        self.valid: int = valid
        self.total: int = total
        self.inference_time_sum: float = 0.0

    def get_average_inference_time(self) -> float:
        return self.inference_time_sum / self.total

    def get_precision(self) -> float:
        return self.valid / self.total


if __name__ == "__main__":
    fix_random_seeds(RAND_SEED)

    # get annotations
    with open(str(data_dir / "ccm_train_annotation.json"), "r") as f:
        dic = json.load(f)
        annotations: List[Dict] = dic["annotations"]
        containers: List[Dict] = dic["containers"]

    # setup model & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskChallenger3()
    # model.load_state_dict(torch.load(current_dir / "20220120-training-2-result.pt"))
    model.load_state_dict(torch.load(current_dir / "20220120-training-2-result.pt"))
    model = model.to(device)
    model.eval()

    # use same validation dataset (picked from `20220120-training-2.py`)
    mv_val2train = 50
    validation_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        mv_val2train=mv_val2train,
        train=False,
        return_also_data_id=True,
    )
    val_dataloader = DataLoader(validation_dataset, specified_seed=RAND_SEED)

    # task1 (level estimation)
    t1_total_estimation = Estimation(0, 0)
    estimations_t1: List[Estimation] = [Estimation(0, 0) for _ in range(len(containers))]
    validation_dataset.query = "level"
    model.task_id = 1
    with torch.no_grad():
        for val_data, label, data_id in val_dataloader:
            # data transport
            if device != torch.device("cpu"):
                val_data = val_data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
            container_type: int = annotations[data_id]["container id"] - 1  # 0-indexed
            start_time = time.process_time()
            pred = model(val_data)
            process_time = time.process_time() - start_time
            estimations_t1[container_type].total += 1
            estimations_t1[container_type].valid += int(pred.argmax(1) == label)
            estimations_t1[container_type].inference_time_sum += process_time
            t1_total_estimation.total += 1
            t1_total_estimation.valid += int(pred.argmax(1) == label)
            t1_total_estimation.inference_time_sum += process_time

    # task2 (type estimation)
    t2_total_estimation = Estimation(0, 0)
    estimations_t2: List[Estimation] = [Estimation(0, 0) for _ in range(len(containers))]
    validation_dataset.query = "type"
    model.task_id = 2
    with torch.no_grad():
        for val_data, label, data_id in val_dataloader:
            # data transport
            if device != torch.device("cpu"):
                val_data = val_data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
            container_type: int = annotations[data_id]["container id"] - 1  # 0-indexed
            start_time = time.process_time()
            pred = model(val_data)
            process_time = time.process_time() - start_time
            estimations_t2[container_type].total += 1
            estimations_t2[container_type].valid += int(pred.argmax(1) == label)
            estimations_t2[container_type].inference_time_sum += process_time
            t2_total_estimation.total += 1
            t2_total_estimation.valid += int(pred.argmax(1) == label)
            t2_total_estimation.inference_time_sum += process_time

    # output
    for container_type in range(len(containers)):
        c_name = containers[container_type]["name"]
        print("")
        print(f"ID: {container_type + 1}; {c_name}  -- Total Num: {estimations_t1[container_type].total}")
        print(f"  On task1 (level estimation)")
        print(f"    precision : {estimations_t1[container_type].get_precision():.3f}", end=" ")
        print(f"| averaged inference time : {estimations_t1[container_type].get_average_inference_time():.3f}")
        print(f"  On task2 (type estimation)")
        print(f"    precision : {estimations_t2[container_type].get_precision():.3f}", end=" ")
        print(f"| averaged inference time : {estimations_t2[container_type].get_average_inference_time():.3f}")

    print("")
    print("")
    assert t1_total_estimation.total == t2_total_estimation.total
    print(f"For all validation data  -- Total Num : {t1_total_estimation.total}")
    print(f"  On task1 (level estimation)")
    print(f"    precision : {t1_total_estimation.get_precision():.3f}", end=" ")
    print(f"| averaged inference time : {t1_total_estimation.get_average_inference_time():.3f}")
    print(f"  On task2 (type estimation)")
    print(f"    precision : {t2_total_estimation.get_precision():.3f}", end=" ")
    print(f"| averaged inference time : {t2_total_estimation.get_average_inference_time():.3f}")

# STDOUT
# ```
# > python 20220120-detailed_validation.py
#
# ID: 1; red cup  -- Total Num: 11
#   On task1 (level estimation)
#     precision : 1.000 | averaged inference time : 0.002
#   On task2 (type estimation)
#     precision : 0.909 | averaged inference time : 0.003
#
# ID: 2; small white cup  -- Total Num: 4
#   On task1 (level estimation)
#     precision : 1.000 | averaged inference time : 0.002
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.002
#
# ID: 3; small transparent cup  -- Total Num: 6
#   On task1 (level estimation)
#     precision : 1.000 | averaged inference time : 0.003
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.003
#
# ID: 4; green glass  -- Total Num: 7
#   On task1 (level estimation)
#     precision : 0.857 | averaged inference time : 0.004
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.002
#
# ID: 5; wine glass  -- Total Num: 12
#   On task1 (level estimation)
#     precision : 1.000 | averaged inference time : 0.002
#   On task2 (type estimation)
#     precision : 0.917 | averaged inference time : 0.002
#
# ID: 6; champagne flute  -- Total Num: 13
#   On task1 (level estimation)
#     precision : 1.000 | averaged inference time : 0.002
#   On task2 (type estimation)
#     precision : 0.923 | averaged inference time : 0.003
#
# ID: 7; cereal box  -- Total Num: 8
#   On task1 (level estimation)
#     precision : 0.875 | averaged inference time : 0.002
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.003
#
# ID: 8; biscuit box  -- Total Num: 9
#   On task1 (level estimation)
#     precision : 0.889 | averaged inference time : 0.003
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.002
#
# ID: 9; tea box  -- Total Num: 8
#   On task1 (level estimation)
#     precision : 0.875 | averaged inference time : 0.041
#   On task2 (type estimation)
#     precision : 1.000 | averaged inference time : 0.004
#
#
# For all validation data  -- Total Num : 78
#   On task1 (level estimation)
#     precision : 0.949 | averaged inference time : 0.006
#   On task2 (type estimation)
#     precision : 0.962 | averaged inference time : 0.003
# ```
