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


class TaskChallenger2(nn.Module):
    def __init__(self, task_id: int = 1):
        super(TaskChallenger2, self).__init__()
        self.task_id = task_id
        self.encoder = LogMelEncoder(num_encoder_blocks=4, num_heads=4)
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
    model = TaskChallenger2()
    # model.load_state_dict(torch.load(current_dir / "20220111-result2.pt"))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.5, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.80)
    mv_val2train = 50
    train_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        mv_val2train=mv_val2train,
        train=True,
    )
    val_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        mv_val2train=mv_val2train,
        train=False,
    )
    train_dataloader = DataLoader(train_dataset, specified_seed=RAND_SEED, shuffle=True)
    val_dataloader = DataLoader(val_dataset, specified_seed=RAND_SEED, shuffle=True)

    train_loss_t1 = []
    val_loss_t1 = []
    train_loss_t2 = []
    val_loss_t2 = []

    min_val_loss_sum_t1_t2: float = float("inf")
    best_loss_pair = (12.34, 45.67)
    best_acc_pair = (0.0, 0.0)

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
        train_loss_t1.append(metrics["train loss"])
        val_loss_t1.append(metrics["val loss"])
        val_acc_t1 = metrics["val accuracy"]
        print(metrics)

        train_dataset.query = "level"
        val_dataset.query = "level"
        train_dataset.random_crop = False
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
        train_loss_t2.append(metrics["train loss"])
        val_loss_t2.append(metrics["val loss"])
        val_acc_t2 = metrics["val accuracy"]
        print(metrics)

        lr_scheduler.step()

        if min_val_loss_sum_t1_t2 > max(0.15, val_loss_t1[-1]) + max(0.15, val_loss_t2[-1]):
            min_val_loss_sum_t1_t2 = val_loss_t1[-1] + val_loss_t2[-1]
            best_loss_pair = val_loss_t1[-1], val_loss_t2[-1]
            best_acc_pair = val_acc_t1, val_acc_t2
            torch.save(model.state_dict(), current_dir / "20220111-result2-alt.pt")

    plt.plot(train_loss_t1, label="train loss: t1")
    plt.plot(val_loss_t1, label="val loss: t1")
    plt.plot(train_loss_t2, label="train loss: t2")
    plt.plot(val_loss_t2, label="val loss: t2")
    plt.legend()
    plt.savefig(str(current_dir / "20220111-result2-alt.png"))

    print(f"best (val_loss_t1, val_loss_t2) pair is {best_loss_pair}!")
    print(f"then (val_acc_t1, val_acc_t2) pair is {best_acc_pair}!")
