from typing import List

import torch

from models import Encoder


class EEGNet(torch.nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        bias: bool,
        n_spectral: int,
        n_spatial: int,
        n_pointwise: int,
        n_channels: int,
        prob_dropout: float,
        hidden_activation: str,
        initialization: str,
    ) -> None:
        super().__init__()

        self.size_input = size_input
        self.size_output = size_output
        self.bias = bias
        self.n_spectral = n_spectral
        self.n_spatial = n_spatial
        self.n_pointwise = n_pointwise
        self.n_channels = n_channels
        self.prob_dropout = prob_dropout
        self.hidden_activation = hidden_activation
        self.initialization = initialization

        self.sequential: torch.nn.Sequential

        self.create_network()

    def create_network(self) -> None:
        sequential: List[torch.nn.Module] = []

        n_features = (self.size_input // 8) // 8

        block1_kernel = (
            128 - 1  # 1/4 of sampling frequency, in paper it's 1/2 but seems too big
        )

        block2_kernel = 16 - 1  # arbitrary

        # Block 1

        sequential.append(
            torch.nn.ReflectionPad2d(
                (
                    (block1_kernel) // 2,
                    block1_kernel - (block1_kernel) // 2,
                    0,
                    0,
                )
            )
        )

        sequential.append(torch.nn.Conv2d(1, self.n_spectral, (1, 128), bias=False))
        sequential.append(
            torch.nn.BatchNorm2d(self.n_spectral, momentum=0.01, eps=0.001)
        )
        sequential.append(
            torch.nn.Conv2d(
                self.n_spectral,
                self.n_pointwise * self.n_spectral,
                (self.n_channels, 1),
                groups=self.n_spectral,
                bias=False,
            )
        )
        sequential.append(
            torch.nn.BatchNorm2d(
                self.n_pointwise * self.n_spectral, momentum=0.01, eps=0.001
            )
        )
        sequential.append(
            torch.nn.ELU() if self.hidden_activation == "elu" else torch.nn.ReLU()
        )
        sequential.append(torch.nn.AvgPool2d((1, 8)))
        sequential.append(torch.nn.Dropout(p=self.prob_dropout))

        # Block 2

        sequential.append(
            torch.nn.ZeroPad2d(
                (
                    (block2_kernel) // 2,
                    block2_kernel - (block2_kernel) // 2,
                    0,
                    0,
                )
            )
        )

        sequential.append(
            torch.nn.Conv2d(
                self.n_pointwise * self.n_spectral,
                self.n_pointwise * self.n_spectral,
                (1, 16),
                groups=self.n_pointwise * self.n_spectral,
                bias=True,
            )
        )
        sequential.append(
            torch.nn.Conv2d(
                self.n_pointwise * self.n_spectral,
                self.n_spatial,
                (1, 1),
                bias=False,
            )
        )
        sequential.append(
            torch.nn.BatchNorm2d(self.n_spatial, momentum=0.01, eps=0.001)
        )
        sequential.append(
            torch.nn.ELU() if self.hidden_activation == "elu" else torch.nn.ReLU()
        )
        sequential.append(torch.nn.AvgPool2d((1, 8)))
        sequential.append(torch.nn.Dropout(p=self.prob_dropout))

        # Fully connected block

        sequential.append(torch.nn.Flatten())
        sequential.append(
            torch.nn.Linear(self.n_spatial * n_features, self.size_output, bias=True)
        )

        self.sequential = torch.nn.Sequential(*sequential)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class MultiEEGNet(Encoder):
    def __init__(
        self,
        size_input: int,
        size_output: int = 768,
        bias: bool = True,
        n_spectral: int = 8,
        n_spatial: int = 16,
        n_pointwise: int = 2,
        n_channels: int = 1,
        prob_dropout: float = 0.2,
        hidden_activation: str = "elu",
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.eegnet = EEGNet(
            size_input,
            size_output,
            bias,
            n_spectral,
            n_spatial,
            n_pointwise,
            n_channels,
            prob_dropout,
            hidden_activation,
            initialization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_vectors = self.eegnet.forward(x.view(-1, 1, 1, x.shape[-1]))
        channel_vectors = channel_vectors.view((*x.shape[:-1], -1))

        return channel_vectors
