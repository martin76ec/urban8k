"""CNN backbone producing (B, channels, freq_bins, T)."""

from __future__ import annotations

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """Convolutional feature extractor for log-Mel spectrograms.

    Input:  (B, 1, M, T)  log-Mel spectrogram.
    Output: (B, channels, freq_bins, T)  downsampled feature maps.

    The frequency axis is reduced to ``freq_bins`` via pooling; the temporal
    axis is preserved so downstream sequence models can attend over T.
    """

    def __init__(self, channels: int = 32, freq_bins: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # reduce freq, keep time
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((freq_bins, None)),  # (B, channels, freq_bins, T)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, M, T).

        Returns:
            (B, channels, freq_bins, T).
        """
        out: torch.Tensor = self.net(x)
        return out
