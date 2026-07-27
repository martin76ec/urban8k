"""Audio feature extraction: log-Mel spectrogram."""

from __future__ import annotations

import torch
import torchaudio


def extract_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """Compute a log-Mel spectrogram from a mono waveform.

    Args:
        waveform: 1D tensor of shape (samples,) or (1, samples).
        sample_rate: sampling rate of the waveform.
        n_mels: number of Mel filters.
        n_fft: FFT window size.
        hop_length: hop between successive STFT frames.

    Returns:
        Tensor of shape (n_mels, T_frames) in log scale.
    """
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-6)
    return log_mel


def normalize_log_mel(log_mel: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardize a log-Mel spectrogram with train-set statistics."""
    return (log_mel - mean[:, None]) / (std[:, None] + 1e-6)


def compute_dataset_stats(
    log_mels: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frequency-bin mean and std from a list of log-Mel spectrograms."""
    all_means = torch.stack([m.mean(dim=1) for m in log_mels], dim=0)
    mean = all_means.mean(dim=0)
    all_vars = torch.stack([m.var(dim=1, unbiased=False) for m in log_mels], dim=0)
    var = all_vars.mean(dim=0)
    std = torch.sqrt(var + 1e-6)
    return mean, std
