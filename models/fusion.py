from typing import Optional

import torch
from torch import nn

from models import Fusion


def normalize(vec: torch.Tensor, complex_unitary: bool = False) -> torch.Tensor:
    result: torch.Tensor

    if complex_unitary:
        fft_vec = torch.fft.rfftn(vec, s=vec.shape[0], dim=0)
        norm_fft = fft_vec.abs().clamp(min=1e-8)
        normed_vec = torch.div(fft_vec, norm_fft)
        result = torch.fft.irfftn(normed_vec, s=vec.shape[0], dim=0)
    else:
        result = torch.nn.functional.normalize(vec, p=1, dim=0)

    return result


def get_fractional_vector(
    seed_vec: torch.Tensor,
    size: int,
    scale: int = 1,
    exponents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if exponents is None:
        exponents = torch.arange(0, size) * scale / size
        exponents = exponents.unsqueeze(0).to(seed_vec.device)
    fft_seed = torch.fft.rfftn(seed_vec, s=seed_vec.shape[0], dim=0)
    pow_vec = fft_seed ** exponents.squeeze()
    result: torch.Tensor = torch.fft.irfftn(pow_vec, s=seed_vec.shape[0], dim=0)

    return result


def fft_convolution(
    signal: torch.Tensor, kernel: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    signal_fft = torch.fft.rfftn(signal, dim=dim)
    kernel_fft = torch.fft.rfftn(kernel, dim=dim)

    conv: torch.Tensor = torch.fft.irfftn(signal_fft * kernel_fft, dim=dim)

    return conv


class CircularEncoding(nn.Module):
    def __init__(
        self,
        size: int = 1000,
    ) -> None:
        super().__init__()

        channels_matrix = torch.normal(mean=0, std=1.0 / size, size=(size, 1))
        self.channels_matrix = nn.Parameter(
            normalize(channels_matrix, complex_unitary=True), requires_grad=False
        )

    def __call__(
        self, sample: torch.Tensor, multiplier: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n_channels = sample.shape[2]

        channels_matrix = self.generate_codebook(multiplier, n_channels)

        encoded_sample = fft_convolution(
            sample.float(),
            channels_matrix.t().unsqueeze(0).unsqueeze(0).float(),
            dim=-1,
        ).sum(dim=2)

        return encoded_sample.to(sample.dtype)

    def generate_codebook(
        self, param: Optional[torch.Tensor], n_channels: int
    ) -> torch.Tensor:
        codebook = get_fractional_vector(
            self.channels_matrix.float(),
            size=n_channels,
            exponents=param,
        )

        return codebook


class CircularEncodingLayer(Fusion):
    def __init__(
        self,
        size: int = 1000,
        patients: int = 1,
        datasets: int = 1,
    ) -> None:
        super().__init__()

        self.encoder = CircularEncoding(size=size)

        self.combination_matrix = nn.Parameter(torch.zeros((datasets, patients, 200)))
        self.init_matrix = self.register_buffer(
            "hrr_init_matrix",
            torch.ones(
                (
                    datasets,
                    patients,
                )
            ),
        )

    def forward(self, x: torch.Tensor, patient: int, dataset: int) -> torch.Tensor:
        n_channels = x.shape[2]
        with torch.inference_mode():
            if self.hrr_init_matrix[dataset, patient]:
                self.combination_matrix.data[dataset, patient, :n_channels] = (
                    torch.arange(0, n_channels) / n_channels
                )
                self.hrr_init_matrix[dataset, patient] = 0
        multiplier = (
            self.combination_matrix[dataset, patient, :n_channels].clamp(min=0)
            / self.combination_matrix[dataset, patient, :n_channels].abs().max()
            + 1.0
        ).unsqueeze(0)
        return self.encoder(x, multiplier)
