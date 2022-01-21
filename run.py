"""get data directory from cli & generate """
import argparse
import torch
import pandas as pd
from pathlib import Path

from task1and2.inference import run as task1and2
from task3.inference import run as task3
from utilities import merge_results

if __name__ == "__main__":
    print("current device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str)
    parser.add_argument("output_path", help="Path to output the resulting csv file e.g.) ./out.csv", type=str)
    parser.add_argument('--draw', default=True, action='store_true', help='Output visual results in ./results')
    parser.add_argument('--validation_test', default=False, action='store_false', help='Output visual results in ./results')
    parser.add_argument(
        "-m",
        "--model_path",
        help='Path to the stored model. Defaults to "./task_challenger3.pt"',
        default="./task_challenger3.pt",
    )
    args = parser.parse_args()

    # prepare paths
    output_path = Path(args.output_path)
    csv_paths = {
        "task1and2": output_path.parents[0] / "out_task1_2.csv",
        "task3": output_path.parents[0] / "out_task3.csv",
    }

    # filling type & level classification
    task1and2(args, csv_paths["task1and2"])

    # container capacity estimation
    task3(args, csv_paths["task3"])

    # merge results
    df = merge_results(csv_paths["task1and2"], csv_paths["task3"])
    df.to_csv(output_path)

    print("Success...!")
