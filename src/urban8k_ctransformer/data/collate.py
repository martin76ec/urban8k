"""Collate function with padding and ``src_key_padding_mask``."""

from __future__ import annotations

import torch


def pad_collate(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of log-Mel spectrograms to a common temporal length.

    Args:
        batch: list of (log_mel (M, T), label, length).

    Returns:
        features: (B, 1, M, T_max)
        labels:   (B,)
        lengths:  (B,)  valid frames per sample
        padding_mask: (B, T_max)  True where padding (for src_key_padding_mask).
    """
    log_mels = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)

    n_mels = log_mels[0].shape[0]
    t_max = int(lengths.max().item())

    features = torch.zeros(len(batch), 1, n_mels, t_max)
    padding_mask = torch.ones(len(batch), t_max, dtype=torch.bool)  # True = pad

    for i, mel in enumerate(log_mels):
        t = mel.shape[1]
        features[i, 0, :, :t] = mel
        padding_mask[i, :t] = False

    return features, labels, lengths, padding_mask
