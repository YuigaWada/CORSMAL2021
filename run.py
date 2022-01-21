"""get data directory from cli & generate """
import argparse
import torch
import pandas as pd
from pathlib import Path

from task1and2.inference import run as task1and2
from task3.inference import run as task3
from utilities import merge_results, print_header

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str)
    parser.add_argument("output_path", help="Path to output the resulting csv file e.g.) ./out.csv", type=str)
    parser.add_argument('--draw', default=True, action='store_true', help='Output visual results in ./output')
    parser.add_argument('--validation_task3', default=False, action='store_false', help='Validation for task3?')
    parser.add_argument(
        "-m",
        "--model_path",
        help='Path to the stored model. Defaults to "./task_challenger3.pt"',
        default="./task_challenger3.pt",
    )
    args = parser.parse_args()

    # print info
    print_header(args, device)

    # prepare paths
    output_path = Path(args.output_path).resolve()
    output_path = output_path if not output_path.is_dir() else output_path / "out.csv"
    csv_paths = {
        "task1and2": output_path.parents[0] / "out_task1_2.csv",
        "task3": output_path.parents[0] / "out_task3.csv",
    }

    # filling type & level classification
    print("\nStarting Task1 and Task2 ...\n")
    task1and2(args, csv_paths["task1and2"])

    # container capacity estimation
    print("\nStarting Task3 ...\n")
    task3(args, csv_paths["task3"])

    # merge results
    df = merge_results(csv_paths["task1and2"], csv_paths["task3"])
    df.to_csv(output_path)

    print("Success...!")
