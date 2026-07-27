"""Prepare data: build a small metadata index and train-set normalization stats.

This script does NOT download UrbanSound8K. It expects the dataset to be present
under ``data.root``. It produces ``data/processed/`` with a metadata pickle and
per-bin mean/std computed from the training folds only.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from urban8k_ctransformer.config import load_config
from urban8k_ctransformer.data import (
    UrbanSound8KDataset,
    load_metadata,
    official_folds_split,
)
from urban8k_ctransformer.utils.logging import get_logger

_logger = get_logger("urban8k.prepare")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare UrbanSound8K metadata and stats")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _logger.info("Loading metadata from %s", cfg.data.root)
    metadata = load_metadata(cfg.data.root)
    split = official_folds_split(metadata)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save split metadata as CSVs for inspection.
    split.train.to_csv(out_dir / "train.csv", index=False)
    split.val.to_csv(out_dir / "val.csv", index=False)
    split.test.to_csv(out_dir / "test.csv", index=False)

    # Compute normalization stats from train fold only (no normalization applied yet).
    _logger.info("Computing log-Mel stats over %d training samples", len(split.train))
    ds = UrbanSound8KDataset(split.train, cfg.data.root, cfg.data)
    log_mels: list[torch.Tensor] = []
    for i in tqdm(range(len(ds)), desc="stats"):
        mel, _label, _len = ds[i]
        log_mels.append(mel)

    all_means = torch.stack([m.mean(dim=1) for m in log_mels], dim=0)
    mean = all_means.mean(dim=0)
    all_vars = torch.stack([m.var(dim=1, unbiased=False) for m in log_mels], dim=0)
    var = all_vars.mean(dim=0)
    std = torch.sqrt(var + 1e-6)

    with (out_dir / "norm_stats.pkl").open("wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    _logger.info("Wrote normalization stats to %s", out_dir / "norm_stats.pkl")
    _logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
