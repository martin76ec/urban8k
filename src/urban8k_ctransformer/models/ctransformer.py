"""CTransformer classifier: CNN backbone + Transformer encoder."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .cnn_backbone import CNNBackbone
from .positional_encoding import PositionalEncoding


class CTransformerClassifier(nn.Module):
    """CNN backbone followed by a Transformer encoder and a linear head.

    Tensor contract:
        input:  (B, 1, M, T)  log-Mel spectrogram
        CNN:    (B, 32, 16, T)
        reshape:(B, T, 512)   # channels × freq_bins
        proj:   (B, T, 256)
        PE:     (B, T, 256)
        encoder:(B, T, 256)
        pool:   (B, 256)      # masked mean over T
        head:   (B, num_classes)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.cnn = CNNBackbone(channels=cfg.cnn_channels, freq_bins=cfg.cnn_frequency_bins)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=cfg.d_model, max_len=cfg.max_len, dropout=cfg.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(
        self, features: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass producing logits.

        Args:
            features: (B, 1, M, T) log-Mel spectrogram.
            src_key_padding_mask: (B, T) boolean mask, True where padding.

        Returns:
            logits: (B, num_classes).
        """
        # (B, 1, M, T) -> (B, C, F, T)
        cnn_out = self.cnn(features)
        b, c, f, t = cnn_out.shape
        # (B, C, F, T) -> (B, T, C*F)
        seq = cnn_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        # Contract check: C*F must equal input_dim.
        if c * f != self.cfg.input_dim:
            raise RuntimeError(
                f"CNN flat dim {c * f} != input_dim {self.cfg.input_dim}; "
                "check cnn_channels * cnn_frequency_bins"
            )
        seq = self.input_proj(seq)  # (B, T, d_model)
        seq = self.pos_encoder(seq)
        seq = self.encoder(seq, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        pooled = self._masked_mean(seq, src_key_padding_mask)  # (B, d_model)
        logits: torch.Tensor = self.classifier(pooled)  # (B, num_classes)
        return logits

    @staticmethod
    def _masked_mean(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over time ignoring padding positions.

        Args:
            x: (B, T, D).
            padding_mask: (B, T) True where padding; None means no padding.
        """
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        summed = (x * valid).sum(dim=1)  # (B, D)
        count = valid.sum(dim=1).clamp(min=1.0)  # (B, 1)
        out: torch.Tensor = summed / count
        return out
