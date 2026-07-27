"""Audio feature extraction: log-Mel spectrogram (torch + soundfile only).

No torchaudio dependency: avoids the torchcodec/CUDA-13 wheel issue on the
H200 server. Audio is read with ``soundfile`` and the log-Mel spectrogram is
computed with pure ``torch`` operations (STFT + Mel filterbank).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch


def _mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None
) -> torch.Tensor:
    """Return a (n_mels, n_fft//2 + 1) Mel filterbank matrix (Slaney-style)."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2.0, n_freqs)

    def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = _hz_to_mel(torch.tensor(f_min))
    mel_max = _hz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(float(mel_min), float(mel_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        for k in range(n_freqs):
            f = fft_freqs[k]
            if f <= left or f >= right:
                continue
            rising = (f - left) / (center - left) if center > left else 1.0
            falling = (right - f) / (right - center) if right > center else 1.0
            fb[m, k] = min(rising, falling)
    # Slaney normalization: area under each filter ~ 1.
    fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-10)
    return fb


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Read an audio file with soundfile and return (mono waveform, sample_rate)."""
    data, sr = sf.read(path, always_2d=True)
    # data: (samples, channels); downmix to mono.
    waveform = torch.from_numpy(data).float().mean(dim=1)
    return waveform, int(sr)


def resample_linear(waveform: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """Linear resampling when input/output rates differ.

    Uses torch.nn.functional.interpolate; adequate for UrbanSound8K where the
    exact resampling filter is not critical (we recompute Mel anyway).
    """
    if sr_in == sr_out:
        return waveform
    ratio = sr_out / sr_in
    target_len = int(round(waveform.shape[0] * ratio))
    out: torch.Tensor = torch.nn.functional.interpolate(
        waveform.unsqueeze(0).unsqueeze(0), size=target_len, mode="linear", align_corners=False
    )
    return out.squeeze(0).squeeze(0)


def extract_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    filterbank: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a log-Mel spectrogram from a mono waveform using torch only.

    Args:
        waveform: 1D tensor of shape (samples,) or (1, samples).
        sample_rate: sampling rate of the waveform.
        n_mels: number of Mel filters.
        n_fft: FFT window size.
        hop_length: hop between successive STFT frames.
        filterbank: optional precomputed (n_mels, n_fft//2+1) Mel filterbank.

    Returns:
        Tensor of shape (n_mels, T_frames) in log scale.
    """
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")

    if filterbank is None:
        filterbank = _mel_filterbank(n_mels, n_fft, sample_rate)

    # Hann window registered on the same device as the waveform.
    window = torch.hann_window(n_fft, periodic=True, device=waveform.device, dtype=waveform.dtype)
    # STFT: returns complex tensor (1, n_fft//2+1, T_frames).
    complex_spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    # Power spectrogram: |STFT|^2.
    spec = complex_spec.abs().pow(2.0)  # (F, T_frames)
    # Apply Mel filterbank: (n_mels, F) @ (F, T) -> (n_mels, T)
    mel = filterbank.to(spec.dtype) @ spec
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


# Keep numpy import used (for type compatibility in helpers).
_ = np