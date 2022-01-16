from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

metrics_t = Dict[str, float]


def classification_loop(
    device: torch.device,
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    enable_amp=False,
) -> Tuple[nn.Module, metrics_t]:
    """run 1 epoch and return model & metrics
    Args:
        device (torch.device): Specify the processing unit.
        model (torch.nn.Module): The model to be trained & validated.
        loss_fn (torch.nn.modules.loss._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_dataloader (torch.utils.data.DataLoader): Data loader of training dataset.
        validation_dataloader (torch.utils.data.DataLoader): Data loader of validation dataset.
        enable_amp (bool, optional): Use automatic mixed precision or not. Defaults to False.
    Returns:
        Tuple[nn.Module, metrics_t]:
            trained model and metrics
    """
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    model.train()
    train_loss_sum = 0
    train_correct_pred = 0
    num_train_data = len(train_dataloader.dataset)  # type: ignore  # map-style Dataset has __len__()
    num_batches_train = len(train_dataloader)
    for data, target in train_dataloader:
        # data transport
        if device != torch.device("cpu"):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        for param in model.parameters():  # fast zero_grad
            param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_amp):
            prediction = model(data)
            loss = loss_fn(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_sum += loss.item()
        train_correct_pred += (prediction.argmax(1) == target).type(torch.float).sum().item()
    train_loss_avg = train_loss_sum / num_batches_train
    train_accuracy = train_correct_pred / num_train_data

    model.eval()
    validation_loss_sum = 0
    validation_correct_pred = 0
    num_validation_data = len(validation_dataloader.dataset)  # type: ignore  # map-style Dataset has __len__()
    num_batches = len(validation_dataloader)
    with torch.no_grad():
        for data, target in validation_dataloader:
            # data transport
            if device != torch.device("cpu"):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                prediction = model(data)
                loss = loss_fn(prediction, target)

            validation_loss_sum += loss.item()
            validation_correct_pred += (prediction.argmax(1) == target).type(torch.float).sum().item()
    val_loss_avg = validation_loss_sum / num_batches
    val_accuracy = validation_correct_pred / num_validation_data

    return model, {
        "train loss": train_loss_avg,
        "train accuracy": train_accuracy,
        "val loss": val_loss_avg,
        "val accuracy": val_accuracy,
    }
