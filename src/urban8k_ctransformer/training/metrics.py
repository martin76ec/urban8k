"""Metric computation (single source of truth)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """Compute accuracy and macro-F1 from a batch of logits.

    Args:
        logits: (N, C) raw logits.
        labels: (N,) ground truth.
        num_classes: number of classes.

    Returns:
        Dict with ``accuracy`` and ``macro_f1``.
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    accuracy = float((preds == labels_np).mean())
    # macro-F1 computed manually to avoid depending on a particular average API.
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels_np == c)).sum())
        fp = int(((preds == c) & (labels_np != c)).sum())
        fn = int(((preds != c) & (labels_np == c)).sum())
        denom = tp + fp
        precision = tp / denom if denom > 0 else 0.0
        denom_r = tp + fn
        recall = tp / denom_r if denom_r > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def classification_report_dict(
    logits: torch.Tensor, labels: torch.Tensor, class_names: list[str] | None = None
) -> dict[str, Any]:
    """Return a scikit-learn classification report as a dict."""
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    report: dict[str, Any] = classification_report(
        labels_np, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    return report


def confusion_matrix(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Return a (C, C) confusion matrix."""
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    cm: np.ndarray = sk_confusion_matrix(labels_np, preds, labels=list(range(num_classes)))
    return cm
