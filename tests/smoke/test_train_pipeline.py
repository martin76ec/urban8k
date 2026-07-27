"""Smoke test: end-to-end training pipeline on a synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from urban8k_ctransformer.config import Config, DataConfig, ModelConfig, TrainConfig
from urban8k_ctransformer.data.collate import pad_collate
from urban8k_ctransformer.models import CTransformerClassifier
from urban8k_ctransformer.training import RunManager, train_one_epoch, validate


class _SyntheticDataset(Dataset):
    def __init__(self, n: int = 12, m: int = 64, t_max: int = 64, n_classes: int = 4) -> None:
        self.n = n
        self.m = m
        self.t_max = t_max
        self.n_classes = n_classes
        self.data = [torch.randn(m, t) for t in [40, 55, 60, 48, 32, 64, 50, 45, 60, 38, 52, 57]]
        self.labels = torch.randint(0, n_classes, (n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self.data[idx], int(self.labels[idx]), self.data[idx].shape[1]


def _make_cfg() -> Config:
    return Config(
        seed=0,
        device="cpu",
        data=DataConfig(
            root="data/raw/UrbanSound8K",
            sample_rate=22050,
            n_mels=128,
            max_seconds=4.0,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        ),
        model=ModelConfig(
            cnn_channels=32,
            cnn_frequency_bins=16,
            input_dim=512,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            num_classes=4,
            max_len=256,
        ),
        train=TrainConfig(
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            amp=False,
            compile=False,
            gradient_clip_norm=1.0,
            early_stopping_patience=10,
            selection_metric="val_macro_f1",
        ),
    )


@pytest.mark.smoke
def test_train_pipeline(tmp_path: Path) -> None:
    cfg = _make_cfg()
    device = torch.device("cpu")
    ds = _SyntheticDataset()
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=pad_collate
    )

    model = CTransformerClassifier(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.CrossEntropyLoss()

    run = RunManager(artifacts_root=tmp_path / "runs", run_id="smoke")
    run.write_config(cfg)
    run.write_metadata(cfg)

    train_metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        num_classes=cfg.model.num_classes,
        amp=False,
        gradient_clip_norm=cfg.train.gradient_clip_norm,
        progress=False,
    )
    val_metrics = validate(
        model, val_loader, criterion, device, cfg.model.num_classes, progress=False
    )

    for m in (train_metrics, val_metrics):
        assert "loss" in m and "accuracy" in m and "macro_f1" in m
        assert torch.isfinite(torch.tensor(m["loss"])).item()
        assert 0.0 <= m["accuracy"] <= 1.0

    run.log_epoch(
        {
            "epoch": 0,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": cfg.train.learning_rate,
            "epoch_seconds": 0.0,
            "vram_mb": 0.0,
        }
    )
    run.save_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=0,
        val_metric=val_metrics["macro_f1"],
        cfg=cfg,
        is_best=True,
    )

    # Files exist
    assert (run.run_dir / "config.yaml").exists()
    assert (run.run_dir / "metadata.json").exists()
    assert (run.run_dir / "metrics.jsonl").exists()
    assert (run.ckpt_dir / "best.pt").exists()

    # Reload checkpoint
    ckpt = run.load_checkpoint(which="best")
    model2 = CTransformerClassifier(cfg.model).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])
    run.write_summary({"best_epoch": 0, "best_val_macro_f1": val_metrics["macro_f1"]})
    assert (run.run_dir / "summary.json").exists()


@pytest.mark.smoke
def test_run_refuses_to_overwrite(tmp_path: Path) -> None:
    RunManager(artifacts_root=tmp_path / "runs", run_id="dup")
    with pytest.raises(RuntimeError, match="already exists"):
        RunManager(artifacts_root=tmp_path / "runs", run_id="dup")
    # Resume works
    RunManager(artifacts_root=tmp_path / "runs", run_id="dup", resume=True)
