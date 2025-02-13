from math import ceil, log2
from typing import List

import torch
from torch import nn

from models import Memory


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.6,
    ) -> None:
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.batchnorm1 = nn.BatchNorm1d(n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.batchnorm2 = nn.BatchNorm1d(n_outputs)

        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.batchnorm1,
            self.chomp1,
            self.elu1,
            self.dropout1,
            self.conv2,
            self.batchnorm2,
            self.chomp2,
            self.elu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class TCN(Memory):
    def __init__(
        self,
        size_input: int,
        size_mem: int,
        size_output: int,
        size_embed: int,
        kernel_size: int = 2,
        dropout: float = 0.6,
    ):
        super(TCN, self).__init__()
        n_layers = ceil(log2((size_mem - 1) / (2 * (2 - 1)) + 1))
        size_channels = [size_embed] * int(n_layers)
        self.size_output = size_output
        layers: List[nn.Module] = []
        num_levels = len(size_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = size_input if i == 0 else size_channels[i - 1]
            out_channels = size_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        layers.append(nn.Flatten())
        layers.append(nn.Linear(size_mem * size_channels[-1], size_output))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
