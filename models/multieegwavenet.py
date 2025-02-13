import torch
from torch import nn

from models import Encoder


class EEGWaveNet(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        n_spectral: int = 32,
        n_channels: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.size_input = size_input
        self.size_output = size_output
        self.n_spectral = n_spectral

        self.temp_conv1 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2
        self.temp_conv2 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2
        self.temp_conv3 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2
        self.temp_conv4 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2
        self.temp_conv5 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2
        self.temp_conv6 = nn.Conv1d(
            n_channels, n_channels, kernel_size=2, stride=2, groups=n_channels
        )  # out = out / 2

        self.chpool1 = nn.Sequential(
            nn.Conv1d(n_channels, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.n_spectral, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
        )

        self.chpool2 = nn.Sequential(
            nn.Conv1d(n_channels, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.n_spectral, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
        )

        self.chpool3 = nn.Sequential(
            nn.Conv1d(n_channels, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.n_spectral, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
        )

        self.chpool4 = nn.Sequential(
            nn.Conv1d(n_channels, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.n_spectral, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
        )

        self.chpool5 = nn.Sequential(
            nn.Conv1d(n_channels, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.n_spectral, self.n_spectral, kernel_size=4, groups=1),
            nn.BatchNorm1d(self.n_spectral),
            nn.LeakyReLU(0.01),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_spectral * 5, self.size_output, bias=True),
        )

    def forward(self, x):

        temp_x = self.temp_conv1(x)
        temp_w1 = self.temp_conv2(temp_x)
        temp_w2 = self.temp_conv3(temp_w1)
        temp_w3 = self.temp_conv4(temp_w2)
        temp_w4 = self.temp_conv5(temp_w3)
        temp_w5 = self.temp_conv6(temp_w4)

        w1 = self.chpool1(temp_w1).mean(dim=(-1))
        w2 = self.chpool2(temp_w2).mean(dim=(-1))
        w3 = self.chpool3(temp_w3).mean(dim=(-1))
        w4 = self.chpool4(temp_w4).mean(dim=(-1))
        w5 = self.chpool5(temp_w5).mean(dim=(-1))

        concat_vector = torch.cat([w1, w2, w3, w4, w5], 1)
        output = self.linear(concat_vector)

        return output


class MultiEEGWaveNet(Encoder):
    def __init__(
        self,
        size_input: int,
        size_output: int = 768,
    ) -> None:
        super().__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.eegnet = EEGWaveNet(size_input, size_output, n_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_vectors = self.eegnet.forward(x.view(-1, 1, x.shape[-1]))
        channel_vectors = channel_vectors.view((*x.shape[:-1], -1))

        return channel_vectors
