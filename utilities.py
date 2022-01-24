import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

"""
csv
"""


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
        writer.writerows(lis)


"""
Merge
"""


def merge_results(paths):
    df = pd.read_csv(paths["task1and2"])
    df_t3 = pd.read_csv(paths["task3"])
    df_t4 = pd.read_csv(paths["task4"])

    for index, row in df.iterrows():
        file_id = row["Configuration ID"]
        
        # task3 and task5
        for name in ["Container capacity","Height","Width at the top","Width at the bottom"]:
            value = df_t3[df_t3["Configuration ID"] == file_id][name]
            if len(value): 
                df.loc[index, name] = value[0]

        # task4
        for name in ["Container mass"]:
            value = df_t4[df_t4["Configuration ID"] == file_id][name]
            if len(value): 
                df.loc[index, name] = value[0]

    return df


"""
Print
"""


def print_header(args, device):
    print("============================================================")
    print('device: ', device)

    print('path2data: ', args.path2data)
    print('output_path: ', args.output_path)
    print('model_path: ', args.model_path, "\n")
    print('draw: ', args.draw)
    print('validation_task3: ', args.validation_task3)
    print("============================================================")
