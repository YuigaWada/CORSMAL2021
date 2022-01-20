"""path resolving"""
import sys
from pathlib import Path

current_dir: Path = Path().cwd().resolve()
project_root: Path = current_dir.parent
data_dir: Path = project_root / "data" / "train"

sys.path.append(str(project_root))


"""main"""
import matplotlib.pyplot as plt  # noqa (E402)
import torch  # noqa (E402)
from torch import nn  # noqa (E402)

from corsmal_challenge.data.data_loader import (  # noqa (E402)
    ReproducibleDataLoader as DataLoader,
)
from corsmal_challenge.data.dataset import AudioDataset  # noqa (E402)
from corsmal_challenge.models.task1_2 import TaskChallenger3  # noqa (E402)
from corsmal_challenge.train.train_val import classification_loop  # noqa (E402)
from corsmal_challenge.utils import fix_random_seeds  # noqa (E402)

RAND_SEED = 0
EPOCH = 100

if __name__ == "__main__":
    fix_random_seeds(RAND_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskChallenger3()
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
            torch.save(model.state_dict(), current_dir / "20220120-training-result.pt")

    plt.plot(train_loss_t1, label="train loss: t1")
    plt.plot(val_loss_t1, label="val loss: t1")
    plt.plot(train_loss_t2, label="train loss: t2")
    plt.plot(val_loss_t2, label="val loss: t2")
    plt.legend()
    plt.savefig(str(current_dir / "20220120-training-result.png"))

    print(f"best (val_loss_t1, val_loss_t2) pair is {best_loss_pair}!")
    print(f"then (val_acc_t1, val_acc_t2) pair is {best_acc_pair}!")

# STDOUT
# ```
# :
# {'train loss': 0.03259559987212404, 'train accuracy': 0.9900826446280991, \
#       'val loss': 0.09255558425203593, 'val accuracy': 0.9743589743589743}
# {'train loss': 0.018783402334247124, 'train accuracy': 0.9917355371900827, \
#       'val loss': 0.5271039578048471, 'val accuracy': 0.8589743589743589}
# {'train loss': 0.041603947514882625, 'train accuracy': 0.9917355371900827, \
#       'val loss': 0.12049947416946265, 'val accuracy': 0.9743589743589743}
# {'train loss': 0.020213941789273077, 'train accuracy': 0.9900826446280991, \
#       'val loss': 0.4169351998993985, 'val accuracy': 0.8846153846153846}
# {'train loss': 0.02943369326331549, 'train accuracy': 0.9900826446280991, \
#       'val loss': 0.18701425280734624, 'val accuracy': 0.9102564102564102}
# {'train loss': 0.015433374435863977, 'train accuracy': 0.9950413223140496, \
#       'val loss': 0.41150730150411025, 'val accuracy': 0.8974358974358975}
# best (val_loss_t1, val_loss_t2) pair is (0.10694323163130694, 0.18056305236659215)!
# then (val_acc_t1, val_acc_t2) pair is (0.9743589743589743, 0.9487179487179487)!
# ```
