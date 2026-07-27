"""Training engine, metrics, and run/checkpoint management."""

from .checkpointing import RunManager
from .engine import train_one_epoch, validate
from .metrics import compute_metrics, confusion_matrix

__all__ = [
    "RunManager",
    "train_one_epoch",
    "validate",
    "compute_metrics",
    "confusion_matrix",
]
