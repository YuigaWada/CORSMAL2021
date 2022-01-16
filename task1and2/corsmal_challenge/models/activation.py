"""contains activation functions"""
import torch
from torch import nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    """
    Squared ReLU layer.
    """

    def __init__(self, inplace=True):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(inputs, inplace=self.inplace)
        return x * x
