"""get data directory from cli & generate """
import time
from pathlib import Path
from typing import Dict, List, Union

import torch

from task1and2.corsmal_challenge.data.audio import load_wav
from task1and2.corsmal_challenge.models.task1_2 import TaskChallenger3
from utilities import create_initialized_row, list2csv

def run(args, output_path):
    # get Path
    data_dir: Path = Path(args.path2data)
    audio_dir: Path = data_dir / "audio"
    model_path: Path = Path(args.task1and2_model_path)

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
            arg_dict["Execution time"] = elapsed_time * 1000
            result_list.append(arg_dict)

            print(" ==> done.")

    list2csv(result_list, output_path)
