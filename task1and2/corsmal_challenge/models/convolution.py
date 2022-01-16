"""contains custom convolution layers"""
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from corsmal_challenge.models.activation import SquaredReLU


class DepthWiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        bias: bool = True,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(DepthWiseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=in_channels * expansion,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.expansion: int = expansion

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, self.weight, self.bias)


class PointWiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super(PointWiseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, self.weight, self.bias)


class InvertedResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
        expansion: int = 6,
    ):
        super(InvertedResBlock, self).__init__()
        self.channels: int = channels
        self.kernel_size: Tuple[int, int] = kernel_size
        self.bias: bool = bias
        self.expansion: int = expansion
        self.padding_size = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        self.bn1 = nn.BatchNorm2d(self.channels)
        self.pconv1 = PointWiseConv2d(
            in_channels=self.channels,
            out_channels=self.channels * self.expansion,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.pconv1.out_channels)
        self.dconv = DepthWiseConv2d(
            in_channels=self.pconv1.out_channels,
            expansion=1,
            kernel_size=self.kernel_size,
            stride=(1, 1),
            bias=False,
            padding=self.padding_size,
        )
        self.bn3 = nn.BatchNorm2d(self.dconv.out_channels)
        self.pconv2 = PointWiseConv2d(
            self.dconv.out_channels,
            self.channels,
            bias=self.bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.bn1(inputs)
        x = F.relu6(x, inplace=True)
        x = self.pconv1(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.dconv(x)
        x = self.bn3(x)
        x = F.relu6(x, inplace=True)
        x = self.pconv2(x)
        res = x + inputs
        return res


def make_inverted_res_stack(
    channels: int,
    kernel_size: Tuple[int, int],
    bias: bool,
    expansion: int,
    layers: int,
):
    layer_stack: List[InvertedResBlock] = []

    for _ in range(layers):
        layer_stack.append(InvertedResBlock(channels, kernel_size, bias, expansion))

    return nn.Sequential(*layer_stack)


class LightCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(LightCNNEncoder, self).__init__()
        self.in_channels = in_channels

        self.squared_relu = SquaredReLU()
        self.bn11 = nn.BatchNorm2d(in_channels)
        self.dconv1 = DepthWiseConv2d(
            in_channels,
            expansion=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.bn12 = nn.BatchNorm2d(self.dconv1.out_channels)
        self.pconv1 = PointWiseConv2d(
            self.dconv1.out_channels,
            self.dconv1.out_channels * 2,
            bias=True,
        )

        self.inverted_res_tower1: nn.Sequential = make_inverted_res_stack(
            self.pconv1.out_channels,
            kernel_size=(3, 3),
            bias=False,
            expansion=6,
            layers=6,
        )

        self.bn21 = nn.BatchNorm2d(self.inverted_res_tower1[-1].channels)
        self.dconv2 = DepthWiseConv2d(
            self.inverted_res_tower1[-1].channels,
            expansion=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.bn22 = nn.BatchNorm2d(self.dconv2.out_channels)
        self.pconv2 = PointWiseConv2d(
            self.dconv2.out_channels,
            self.dconv2.out_channels * 4,
            bias=True,
        )

        self.inverted_res_tower2: nn.Sequential = make_inverted_res_stack(
            self.pconv2.out_channels,
            kernel_size=(3, 3),
            bias=False,
            expansion=6,
            layers=6,
        )

        self.bn31 = nn.BatchNorm2d(self.inverted_res_tower2[-1].channels)
        self.dconv3 = DepthWiseConv2d(
            self.inverted_res_tower2[-1].channels,
            expansion=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.bn32 = nn.BatchNorm2d(self.dconv3.out_channels)
        self.pconv3 = PointWiseConv2d(
            self.dconv3.out_channels,
            self.dconv3.out_channels * 4,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            inputs (torch.Tensor): (batches, channels, freq, times)

        Returns:
            torch.Tensor: (batches channels, 1, feature_dim)
        """
        x = self.bn11(inputs)
        x = self.dconv1(x)
        x = self.bn12(x)
        x = self.squared_relu(x)
        x = self.pconv1(x)
        x = self.inverted_res_tower1(x)
        x = self.bn21(x)
        x = self.squared_relu(x)
        x = self.dconv2(x)
        x = self.bn22(x)
        x = self.squared_relu(x)
        x = self.pconv2(x)
        x = self.inverted_res_tower2(x)
        x = self.bn31(x)
        x = self.squared_relu(x)
        x = self.dconv3(x)
        x = self.bn32(x)
        x = self.squared_relu(x)
        x = self.pconv3(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, dim=1)
        return x
