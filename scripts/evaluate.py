"""Evaluate a trained run on the held-out test fold."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from urban8k_ctransformer.config import load_config
from urban8k_ctransformer.data import (
    UrbanSound8KDataset,
    load_metadata,
    official_folds_split,
    pad_collate,
)
from urban8k_ctransformer.models import CTransformerClassifier
from urban8k_ctransformer.training import RunManager, collect_predictions
from urban8k_ctransformer.training.metrics import classification_report_dict, confusion_matrix
from urban8k_ctransformer.utils import resolve_device, seed_everything
from urban8k_ctransformer.utils.logging import get_logger

_logger = get_logger("urban8k.evaluate")

URBAN_SOUND_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def _load_norm_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if not path.exists():
        return None, None
    with path.open("rb") as f:
        d = pickle.load(f)  # noqa: S301
    return d["mean"], d["std"]


def _plot_learning_curves(run: RunManager, summary: dict) -> None:
    rows = run.read_metrics()
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    val_f1 = [r["val_macro_f1"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, val_f1, label="val macro-F1", color="green")
    axes[1].set_title("Val macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run.figures_dir / "learning_curves.png", dpi=120)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, run: RunManager) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=URBAN_SOUND_CLASSES,
        yticklabels=URBAN_SOUND_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(run.figures_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    np.savetxt(run.figures_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a run on the test fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    run = RunManager(run_id=args.run_id, resume=True)
    # Attach file logging (also tees stdout/stderr into run.log).
    global _logger
    _logger = run.attach_file_logger("urban8k.evaluate")
    ckpt = run.load_checkpoint(which=args.checkpoint)
    model = CTransformerClassifier(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    _logger.info("Loaded checkpoint (epoch %s, val_metric %.4f)", ckpt["epoch"], ckpt["val_metric"])

    metadata = load_metadata(cfg.data.root)
    split = official_folds_split(metadata)
    mean, std = _load_norm_stats(Path("data/processed/norm_stats.pkl"))
    test_ds = UrbanSound8KDataset(split.test, cfg.data.root, cfg.data, mean, std)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    logits, labels = collect_predictions(model, test_loader, device)
    num_classes = cfg.model.num_classes
    from urban8k_ctransformer.training.metrics import compute_metrics

    base = compute_metrics(logits, labels, num_classes)
    report = classification_report_dict(logits, labels, URBAN_SOUND_CLASSES)
    cm = confusion_matrix(logits, labels, num_classes)

    _plot_learning_curves(run, {})
    _plot_confusion(cm, run)

    summary = {
        "test_accuracy": base["accuracy"],
        "test_macro_f1": base["macro_f1"],
        "test_classification_report": report,
        "checkpoint_used": args.checkpoint,
        "n_test_samples": int(labels.shape[0]),
    }
    run.write_summary(summary)
    _logger.info("Test accuracy=%.4f macro_f1=%.4f", base["accuracy"], base["macro_f1"])
    print(
        json.dumps({"test_accuracy": base["accuracy"], "test_macro_f1": base["macro_f1"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
