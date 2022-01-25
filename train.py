"""get data directory from cli & generate """
import argparse
import torch
from pathlib import Path

from task1and2.train import train as trainTask1and2
from task3.inference import run as task3
from task4.train import train as trainTask4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str) #we need args.path2data / ccm_train_annotation.json
    parser.add_argument(
        "--task1and2",
        help='To train task1 and task2',
        action='store_true'
    )
    parser.add_argument(
        "--task4",
        help='To training task4',
        action='store_true'
    )

    args = parser.parse_args() 
    args.validation_task3 = False

    # prepare paths
    output_path = Path("./").resolve()
    output_path = output_path if not output_path.is_dir() else output_path / "out.csv"
    csv_paths = {
        "task3": output_path.parents[0] / "out_task3.csv"
    }

    if args.task1and2:
        # train task1 and task2
        print("\nTraining Task1 and Task2 ...\n")
        trainTask1and2(args)

        print("task1and2: Success...! (./train4.pt)")


    if args.task4:
        # container capacity estimation (for training task4)
        print("\nStarting Task3 and Task5...\n")
        task3(args, csv_paths["task3"])

        # train task4
        print("\nTraining Task4 ...\n")
        trainTask4(args, path_for_task3=csv_paths["task3"])

        print("task4: Success...! (./train4.pt)")
