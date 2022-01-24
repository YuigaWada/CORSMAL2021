import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

"""
csv
"""

name_type_table = {
    "Configuration ID": int,
    "Container capacity": float,
    "Container mass": float,
    "Filling mass": int,
    "None": float,
    "Pasta": float,
    "Rice": float,
    "Water": float,
    "Filling type": int,
    "Empty": float,
    "Half-full": float,
    "Full": float,
    "Filling level": int,
    "Width at the top": float,
    "Width at the bottom": float,
    "Height": float,
    "Object safety": int,
    "Distance": int,
    "Angle difference": int,
    "Execution time": float
}


def create_initialized_row() -> Dict[str, Union[int, float]]:
    arg_dict: Dict[str, Union[int, float]] = {
        "Configuration ID": -1,
        "Container capacity": -1,
        "Container mass": -1,
        "Filling mass": -1,
        "None": -1,
        "Pasta": -1,
        "Rice": -1,
        "Water": -1,
        "Filling type": -1,
        "Empty": -1,
        "Half-full": -1,
        "Full": -1,
        "Filling level": -1,
        "Width at the top": -1,
        "Width at the bottom": -1,
        "Height": -1,
        "Object safety": -1,
        "Distance": -1,
        "Angle difference": -1,
        "Execution time": -1,
    }
    return arg_dict


def list2csv(lis: List[Dict[str, Union[int, float]]], path: Path) -> None:
    result = []
    for i in range(len(lis)):
        dt = {}
        for key, value in lis[i].items():
            tp = name_type_table[key]
            if tp is float:
                dt[key] = "{:.5f}".format(value)
            elif tp is int:
                dt[key] = str(value)
            else:
                dt[key] = value
        result.append(dt)

    with open(str(path), "w") as f:
        writer = csv.DictWriter(
            f,
            [
                "Configuration ID",
                "Container capacity",
                "Container mass",
                "Filling mass",
                "None",
                "Pasta",
                "Rice",
                "Water",
                "Filling type",
                "Empty",
                "Half-full",
                "Full",
                "Filling level",
                "Width at the top",
                "Width at the bottom",
                "Height",
                "Object safety",
                "Distance",
                "Angle difference",
                "Execution time",
            ],
        )
        writer.writeheader()
        writer.writerows(result)


"""
Merge
"""


def merge_results(paths):
    row_dict = {}
    for task_name in ["task1and2", "task3", "task4"]:
        with open(str(paths[task_name]), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["Configuration ID"])
                if cid not in row_dict: row_dict[cid] = []
                row_dict[cid].append(row)

    targets = ["Container capacity", "Height", "Width at the top", "Width at the bottom", "Container mass"]
    result = []

    for key in sorted(row_dict.keys()):
        merged = {}
        for row in row_dict[key]:
            for k, v in row.items():
                tp = name_type_table[k]
                v = tp(v)
                if k not in merged:
                    merged[k] = v
                elif k in targets and v != -1:
                    merged[k] = v
                elif k == "Execution time" and v != -1:
                    merged[k] += v

        result.append(merged)

    return result


"""
Print
"""


def print_header(args, device):
    print("============================================================")
    print('device: ', device)

    print('path2data: ', args.path2data)
    print('output_path: ', args.output_path)
    print('task1and2_model_path: ', args.task1and2_model_path)
    print('task4_model_path: ', args.task4_model_path, "\n")
    print('draw: ', args.draw)
    print('validation_task3: ', args.validation_task3)
    print("============================================================")
