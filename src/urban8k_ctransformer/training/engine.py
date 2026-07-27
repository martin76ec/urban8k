"""Training engine: one epoch and validation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import get_logger
from .metrics import compute_metrics

_logger = get_logger("urban8k.train")

# A DataLoader that yields (features, labels, lengths, padding_mask) tuples.
BatchLoader = DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def train_one_epoch(
    model: nn.Module,
    loader: BatchLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp: bool = False,
    gradient_clip_norm: float | None = None,
    scheduler: Any | None = None,
    progress: bool = True,
) -> dict[str, float]:
    """Run a single training epoch and return aggregated metrics."""
    model.train()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    n_batches = 0

    autocast_device = "cuda" if (amp and device.type == "cuda") else "cpu"
    scaler: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
        enabled=amp and device.type == "cuda"
    )

    iterator = tqdm(loader, desc="train", disable=not progress)
    for features, labels, _lengths, padding_mask in iterator:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=amp):
            logits = model(features, src_key_padding_mask=padding_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits_cat, labels_cat, num_classes)
    metrics["loss"] = avg_loss
    return metrics


def validate(
    model: nn.Module,
    loader: BatchLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    progress: bool = True,
) -> dict[str, float]:
    """Run validation and return aggregated metrics."""
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    n_batches = 0

    iterator = tqdm(loader, desc="val", disable=not progress)
    with torch.no_grad():
        for features, labels, _lengths, padding_mask in iterator:
            features = features.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            logits = model(features, src_key_padding_mask=padding_mask)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits_cat, labels_cat, num_classes)
    metrics["loss"] = avg_loss
    return metrics


def collect_predictions(
    model: nn.Module,
    loader: BatchLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) for the whole loader (used by evaluate)."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels, _lengths, padding_mask in loader:
            features = features.to(device)
            padding_mask = padding_mask.to(device)
            logits = model(features, src_key_padding_mask=padding_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


__all__ = ["train_one_epoch", "validate", "collect_predictions"]
