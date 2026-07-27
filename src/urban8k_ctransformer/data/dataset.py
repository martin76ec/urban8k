"""UrbanSound8K dataset with on-the-fly log-Mel extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from ..config import DataConfig
from .features import extract_log_mel


class UrbanSound8KDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Dataset that loads audio slices and computes log-Mel spectrograms.

    Args:
        metadata: DataFrame with columns ``slice_file_name``, ``fold``, ``classID``.
        root: path to the UrbanSound8K root (containing ``audio/``).
        cfg: data configuration block.
        normalize_mean: optional per-bin mean for standardization.
        normalize_std: optional per-bin std for standardization.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | Path,
        cfg: DataConfig,
        normalize_mean: torch.Tensor | None = None,
        normalize_std: torch.Tensor | None = None,
    ) -> None:
        if "slice_file_name" not in metadata.columns or "classID" not in metadata.columns:
            raise ValueError("metadata must have 'slice_file_name' and 'classID' columns")
        self.metadata = metadata.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_audio(self, fold: int, name: str) -> torch.Tensor:
        audio_path = self.root / "audio" / f"fold{fold}" / name
        waveform, sr = torchaudio.load(str(audio_path))
        # to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample
        if sr != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            waveform = resampler(waveform)
        out: torch.Tensor = waveform.squeeze(0)
        return out

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        target = self.cfg.max_samples
        if waveform.shape[0] >= target:
            return waveform[:target]
        pad = target - waveform.shape[0]
        out: torch.Tensor = torch.nn.functional.pad(waveform, (0, pad))
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row = self.metadata.iloc[idx]
        waveform = self._load_audio(int(row["fold"]), str(row["slice_file_name"]))
        waveform = self._fix_length(waveform)
        log_mel = extract_log_mel(waveform, self.cfg.sample_rate, n_mels=self.cfg.n_mels)
        if self.normalize_mean is not None and self.normalize_std is not None:
            log_mel = (log_mel - self.normalize_mean[:, None]) / (
                self.normalize_std[:, None] + 1e-6
            )
        label = int(row["classID"])
        length = log_mel.shape[1]
        return log_mel, label, length
