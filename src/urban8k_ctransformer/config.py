"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: str
    sample_rate: int
    n_mels: int
    max_seconds: float
    num_workers: int
    persistent_workers: bool
    pin_memory: bool

    @property
    def max_samples(self) -> int:
        return int(self.max_seconds * self.sample_rate)


@dataclass(frozen=True)
class ModelConfig:
    cnn_channels: int
    cnn_frequency_bins: int
    input_dim: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    num_classes: int
    max_len: int

    def validate(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {self.num_classes}")
        # Contract: CNN output channels × frequency bins == input_dim of the projection.
        cnn_flat = self.cnn_channels * self.cnn_frequency_bins
        if cnn_flat != self.input_dim:
            raise ValueError(
                f"input_dim ({self.input_dim}) must equal "
                f"cnn_channels * cnn_frequency_bins ({cnn_flat})"
            )


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    amp: bool
    compile: bool
    gradient_clip_norm: float
    early_stopping_patience: int
    selection_metric: str


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    config_path: str = field(default="")

    def validate(self) -> None:
        self.model.validate()


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(
        seed=raw["seed"],
        device=raw["device"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        config_path=str(path),
    )
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return {
        "seed": cfg.seed,
        "device": cfg.device,
        "data": cfg.data.__dict__,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
        "config_path": cfg.config_path,
    }
