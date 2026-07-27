"""Train the CTransformer model on UrbanSound8K."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from urban8k_ctransformer.config import load_config
from urban8k_ctransformer.data import (
    UrbanSound8KDataset,
    load_metadata,
    official_folds_split,
    pad_collate,
)
from urban8k_ctransformer.models import CTransformerClassifier
from urban8k_ctransformer.training import RunManager, train_one_epoch, validate
from urban8k_ctransformer.utils import resolve_device, seed_everything
from urban8k_ctransformer.utils.logging import get_logger

_logger = get_logger("urban8k.train")


def _load_norm_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if not path.exists():
        return None, None
    with path.open("rb") as f:
        d = pickle.load(f)  # noqa: S301
    return d["mean"], d["std"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CTransformer on UrbanSound8K")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume an existing run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    _logger.info("Using device: %s", device)

    run = RunManager(run_id=args.run_id, resume=args.resume)
    run.write_config(cfg)
    run.write_metadata(cfg)
    _logger.info("Run directory: %s", run.run_dir)

    # Data
    metadata = load_metadata(cfg.data.root)
    split = official_folds_split(metadata)
    mean, std = _load_norm_stats(Path("data/processed/norm_stats.pkl"))

    train_ds = UrbanSound8KDataset(split.train, cfg.data.root, cfg.data, mean, std)
    val_ds = UrbanSound8KDataset(split.val, cfg.data.root, cfg.data, mean, std)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        pin_memory=cfg.data.pin_memory,
    )

    model = CTransformerClassifier(cfg.model).to(device)
    if cfg.train.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val = -float("inf")
    if args.resume:
        ckpt = run.load_checkpoint(which="last")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["val_metric"])
        _logger.info(
            "Resumed from epoch %d (best %s=%.4f)",
            start_epoch,
            cfg.train.selection_metric,
            best_val,
        )

    patience_left = cfg.train.early_stopping_patience
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes=cfg.model.num_classes,
            amp=cfg.train.amp and device.type == "cuda",
            gradient_clip_norm=cfg.train.gradient_clip_norm,
        )
        val_metrics = validate(model, val_loader, criterion, device, cfg.model.num_classes)
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": dt,
            "vram_mb": vram,
        }
        run.log_epoch(row)
        # Map selection_metric name (e.g. "val_macro_f1") to the metric key in val_metrics.
        sel_key = cfg.train.selection_metric.split("val_")[-1]
        val_metric = val_metrics[sel_key]
        is_best = val_metric > best_val
        if is_best:
            best_val = val_metric
            patience_left = cfg.train.early_stopping_patience
        else:
            patience_left -= 1
        run.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            val_metric=val_metric,
            cfg=cfg,
            is_best=is_best,
        )
        _logger.info(
            "epoch %d | train_loss=%.4f val_loss=%.4f val_%s=%.4f best=%.4f %s",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            sel_key,
            val_metric,
            best_val,
            "(best)" if is_best else f"(patience {patience_left})",
        )
        if patience_left <= 0:
            _logger.info("Early stopping at epoch %d", epoch)
            break

    best_val_final, best_epoch = run.best_metric(cfg.train.selection_metric)
    run.write_summary(
        {
            "best_epoch": best_epoch,
            f"best_{cfg.train.selection_metric}": best_val_final,
            "total_epochs_trained": epoch + 1,
        }
    )
    _logger.info(
        "Training complete. Best %s=%.4f at epoch %d",
        cfg.train.selection_metric,
        best_val_final,
        best_epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
