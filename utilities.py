import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

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

def merge_results(task1and2_path, task3_path):
    df = pd.read_csv(task1and2_path)
    df_t3 = pd.read_csv(task3_path)

    for index, row in df.iterrows():
        file_id = row["Configuration ID"]
        capacity = df_t3[df_t3["Configuration ID"] == file_id]["Container capacity"]
        if len(capacity) > 0:
            df.loc[index, "Container capacity"] = capacity[0]

    return df