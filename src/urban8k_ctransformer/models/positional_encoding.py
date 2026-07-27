"""Sinusoidal positional encoding registered as a buffer."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (no trainable parameters).

    Args:
        d_model: feature dimension.
        max_len: maximum sequence length supported.
        dropout: dropout applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: (B, T, d_model).

        Returns:
            (B, T, d_model) with positional encoding added.
        """
        t = int(x.shape[1])
        pe = self.pe
        if not isinstance(pe, torch.Tensor):
            raise RuntimeError("positional encoding buffer is not a Tensor")
        max_len = int(pe.shape[1])
        if t > max_len:
            raise ValueError(
                f"sequence length {t} exceeds max_len {max_len}; "
                "increase model.max_len in the config"
            )
        out: torch.Tensor = self.dropout(x + pe[:, :t, :])
        return out
