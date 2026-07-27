"""Splits following UrbanSound8K official folds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitInfo:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_metadata(root: str | Path) -> pd.DataFrame:
    """Load UrbanSound8K.csv metadata."""
    csv_path = Path(root) / "UrbanSound8K.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    return pd.read_csv(csv_path)


def official_folds_split(
    metadata: pd.DataFrame,
    train_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    val_folds: tuple[int, ...] = (9,),
    test_folds: tuple[int, ...] = (10,),
) -> SplitInfo:
    """Build train/val/test splits from official UrbanSound8K folds.

    Args:
        metadata: full UrbanSound8K.csv as a DataFrame.
        train_folds: folds used for training.
        val_folds: folds used for validation / model selection.
        test_folds: folds held out for final evaluation only.
    """
    all_folds = set(train_folds) | set(val_folds) | set(test_folds)
    if len(all_folds) != len(train_folds) + len(val_folds) + len(test_folds):
        raise ValueError("Folds overlap between train, val and test")
    if not all(1 <= f <= 10 for f in all_folds):
        raise ValueError("Folds must be in 1..10")

    train = metadata[metadata["fold"].isin(train_folds)].reset_index(drop=True)
    val = metadata[metadata["fold"].isin(val_folds)].reset_index(drop=True)
    test = metadata[metadata["fold"].isin(test_folds)].reset_index(drop=True)
    return SplitInfo(train=train, val=val, test=test)
