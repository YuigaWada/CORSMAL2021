"""get data directory from cli & generate """
import argparse
import csv
import time
from pathlib import Path
from task1and2.inference import run as task1and2
from task3.inference import run as task3

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str)
    parser.add_argument("output_path", help="Path to output the resulting csv file e.g.) ./out.csv", type=str)
    parser.add_argument('--draw', default=True, action='store_true', help='Output visual results in ./results')
    parser.add_argument('--validation_test', default=False, action='store_false', help='Output visual results in ./results') # todo: taskごとチェック and false??
    # parser.add_argument('--task1and2', default=True, action='store_true', help='Run task1 and task2')
    # parser.add_argument('--task3', default=True, action='store_true', help='Run task3')
    
    parser.add_argument(
        "-m",
        "--model_path",
        help='Path to the stored model. Defaults to "./task_challenger3.pt"',
        default="./task_challenger3.pt",
    )
    args = parser.parse_args()

    # prepare paths
    output_path = Path(args.output_path)
    path_for_task1and2 = output_path.parents[0] / "out_task1_2.csv"
    path_for_task3 = output_path.parents[0] / "out_task3.csv"

    # filling type & level classification
    task1and2(args, path_for_task1and2)

    # container capacity estimation
    task3(args, path_for_task3)

    # merge results


