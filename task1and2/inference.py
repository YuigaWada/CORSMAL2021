"""get data directory from cli & generate """
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Union

import torch

from corsmal_challenge.data.audio import load_wav
from corsmal_challenge.models.task1_2 import TaskChallenger3


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


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help="Path to the given data directory", type=str)
    parser.add_argument("output_path", help="Path to output the resulting csv file e.g.) ./out.csv", type=str)
    parser.add_argument(
        "-m",
        "--model_path",
        help='Path to the stored model. Defaults to "./task_challenger3.pt"',
        default="./task_challenger3.pt",
    )
    args = parser.parse_args()

    # get Path
    data_dir: Path = Path(args.path2data)
    audio_dir: Path = data_dir / "audio"
    output_path: Path = Path(args.output_path)
    model_path: Path = Path(args.model_path)

    # load model & send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskChallenger3()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # inference
    result_list: List[Dict[str, Union[int, float]]] = []
    for wav_file_path in sorted(list(audio_dir.glob("*.wav"))):
        with torch.no_grad():
            arg_dict = create_initialized_row()
            print(f"processing {wav_file_path}", end="")
            arg_dict["Configuration ID"] = int(wav_file_path.stem)
            spectrogram = load_wav(wav_file_path).generate_mel_spectrogram().log2()
            spectrogram = spectrogram.unsqueeze(0).permute(0, 1, 3, 2)  # (batch, channel, sequence, embed_dim)

            # data transport
            if device != torch.device("cpu"):
                spectrogram = spectrogram.to(device, non_blocking=True)

            start_time = time.process_time()

            # task2
            model.task_id = 2
            prediction_t2: torch.Tensor = model(spectrogram).softmax(dim=-1)

            # task1
            model.task_id = 1
            prediction_t1: torch.Tensor = model(spectrogram).softmax(dim=-1)

            elapsed_time = time.process_time() - start_time

            # print(f"prediction_t2: {prediction_t2}")
            # print(f"prediction_t1: {prediction_t1}")
            # print(f"elapsed_time: {elapsed_time:.3g}")

            arg_dict["None"] = float(prediction_t2[0][0])
            arg_dict["Pasta"] = float(prediction_t2[0][1])
            arg_dict["Rice"] = float(prediction_t2[0][2])
            arg_dict["Water"] = float(prediction_t2[0][3])
            arg_dict["Filling type"] = int(torch.argmax(prediction_t2, dim=-1))
            arg_dict["Empty"] = float(prediction_t1[0][0])
            arg_dict["Half-full"] = float(prediction_t1[0][1])
            arg_dict["Full"] = float(prediction_t1[0][2])
            arg_dict["Filling level"] = int(torch.argmax(prediction_t1, dim=-1))
            arg_dict["Execution time"] = elapsed_time
            result_list.append(arg_dict)

            print(" ==> done.")

    list2csv(result_list, output_path)
