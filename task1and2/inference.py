import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Union

from corsmal_challenge.data.audio import load_wav


def create_initial_row() -> Dict[str, Optional[Union[int, float]]]:
    arg_dict = {
        "Configuration ID": None,
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


if __name__ == "__main__":
    current_dir: Path = pathlib.Path().cwd().resolve()
    audio_data_path: Path = current_dir / "data" / "test_pub" / "audio"
    output_path: Path = current_dir

    result_list: List[Dict[str, Optional[Union[int, float]]]] = list()

    num_data = 227
    for i in range(num_data):
        spectrogram = load_wav(audio_data_path / str(i).zfill(6) + ".wav")
        print("TODO : impl inference!")
