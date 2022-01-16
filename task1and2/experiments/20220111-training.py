"""path resolving"""
import sys
from pathlib import Path

current_dir: Path = Path().cwd().resolve()
project_root: Path = current_dir.parent
data_dir: Path = project_root / "data" / "train"

sys.path.append(str(project_root))


"""main"""
from typing import Dict  # noqa (E402)

import matplotlib.pyplot as plt  # noqa (E402)
import torch  # noqa (E402)
from torch import nn  # noqa (E402)
from torch.nn import functional as F  # noqa (E402)

from corsmal_challenge.data.data_loader import (  # noqa (E402)
    ReproducibleDataLoader as DataLoader,
)
from corsmal_challenge.data.dataset import AudioDataset  # noqa (E402)
from corsmal_challenge.models.audio import LogMelEncoder  # noqa (E402)
from corsmal_challenge.models.task1_2 import T1Head, T2Head  # noqa (E402)
from corsmal_challenge.train.train_val import classification_loop  # noqa (E402)
from corsmal_challenge.utils import fix_random_seeds  # noqa (E402)


class TaskChallenger(nn.Module):
    def __init__(self, task_id: int = 1):
        super(TaskChallenger, self).__init__()
        self.task_id = task_id
        self.encoder = LogMelEncoder()
        self.classify_head1 = T1Head()
        self.classify_head2 = T2Head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder(inputs)
        if self.task_id == 1:
            x = self.classify_head1(x[:, 0, :])  # extract embedding of class token
        elif self.task_id == 2:
            x = self.classify_head2(x[:, 0, :])  # extract embedding of class token
        x = x.squeeze(1)
        return x


RAND_SEED = 0
EPOCH = 150

if __name__ == "__main__":
    fix_random_seeds(RAND_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskChallenger()
    model.load_state_dict(torch.load(current_dir / "20220111-result.pt"))
    model = TaskChallenger()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        train=True,
        query="type",
        random_crop=True,
        strong_crop=True,
    )
    val_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        train=False,
        query="type",
        random_crop=False,
        strong_crop=False,
    )
    train_dataloader = DataLoader(train_dataset, specified_seed=RAND_SEED, shuffle=True)
    val_dataloader = DataLoader(val_dataset, specified_seed=RAND_SEED, shuffle=True)

    train_loss_t1 = []
    val_loss_t1 = []
    train_loss_t2 = []
    val_loss_t2 = []

    for step in range(EPOCH):
        train_dataset.query = "type"
        val_dataset.query = "type"
        train_dataset.random_crop = True
        train_dataset.strong_crop = True
        model.task_id = 2

        tup = classification_loop(  # type: ignore
            device,
            model,
            loss_fn,
            optimizer,
            train_dataloader,
            val_dataloader,
            enable_amp=True,
        )
        metrics = tup[1]
        train_loss_t2.append(metrics["train loss"])
        val_loss_t2.append(metrics["val loss"])
        print(metrics)

        train_dataset.query = "level"
        val_dataset.query = "level"
        train_dataset.random_crop = True
        train_dataset.strong_crop = False
        model.task_id = 1

        tup = classification_loop(  # type: ignore
            device,
            model,
            loss_fn,
            optimizer,
            train_dataloader,
            val_dataloader,
            enable_amp=True,
        )
        metrics = tup[1]
        train_loss_t1.append(metrics["train loss"])
        val_loss_t1.append(metrics["val loss"])
        print(metrics)

    plt.plot(train_loss_t1, label="train_loss_t1")
    plt.plot(val_loss_t1, label="val_loss_t1")
    plt.plot(train_loss_t2, label="train_loss_t2")
    plt.plot(val_loss_t2, label="val_loss_t2")
    plt.legend()
    plt.savefig(str(current_dir / "20220111-result.png"))
    torch.save(model.state_dict(), current_dir / "20220111-result.pt")
