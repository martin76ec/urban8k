"""Run/checkpoint management: the only module that knows artifacts/runs/ layout."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import Config, config_to_dict

JSONValue = Any


class RunManager:
    """Manage a single self-contained run directory under ``artifacts/runs/``.

    Layout::

        artifacts/runs/<run_id>/
            config.yaml
            metadata.json
            metrics.jsonl
            summary.json
            checkpoints/best.pt
            checkpoints/last.pt
            figures/
    """

    def __init__(
        self,
        artifacts_root: str | Path = "artifacts/runs",
        run_id: str | None = None,
        resume: bool = False,
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.run_id = run_id or self._default_run_id()
        self.run_dir = self.artifacts_root / self.run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.figures_dir = self.run_dir / "figures"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.yaml"
        self.metadata_path = self.run_dir / "metadata.json"

        if self.run_dir.exists() and not resume:
            raise RuntimeError(
                f"Run '{self.run_id}' already exists at {self.run_dir}. "
                "Use --resume to continue it."
            )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_run_id() -> str:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return ts

    def write_config(self, cfg: Config) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)

    def write_metadata(self, cfg: Config) -> None:
        meta: dict[str, JSONValue] = {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "seed": cfg.seed,
            "config_path": cfg.config_path,
        }
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def log_epoch(self, metrics: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_checkpoint(
        self,
        model_state: dict[str, JSONValue],
        optimizer_state: dict[str, JSONValue],
        epoch: int,
        val_metric: float,
        cfg: Config,
        is_best: bool,
        scheduler_state: dict[str, JSONValue] | None = None,
    ) -> None:
        ckpt: dict[str, JSONValue] = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "epoch": epoch,
            "val_metric": val_metric,
            "config": config_to_dict(cfg),
        }
        # RNG state for resumability.
        ckpt["torch_rng_state"] = torch.get_rng_state().numpy().tolist()
        if torch.cuda.is_available():
            ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        else:
            ckpt["cuda_rng_state"] = None

        last_path = self.ckpt_dir / "last.pt"
        torch.save(ckpt, last_path)
        if is_best:
            shutil.copy2(last_path, self.ckpt_dir / "best.pt")

    def load_checkpoint(self, which: str = "best") -> dict[str, JSONValue]:
        name = "best.pt" if which == "best" else "last.pt"
        path = self.ckpt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} not found")
        loaded: dict[str, JSONValue] = torch.load(
            path, map_location="cpu", weights_only=False
        )
        return loaded

    def write_summary(self, summary: dict[str, JSONValue]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    def read_metrics(self) -> list[dict[str, JSONValue]]:
        if not self.metrics_path.exists():
            return []
        out = []
        with self.metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def best_metric(self, selection_metric: str) -> tuple[float, int]:
        """Return (best_value, best_epoch) from logged metrics."""
        rows = self.read_metrics()
        if not rows:
            raise RuntimeError(f"No metrics logged in {self.metrics_path}")
        key = selection_metric
        best_val = -float("inf")
        best_epoch = -1
        for row in rows:
            if key in row and row[key] > best_val:
                best_val = row[key]
                best_epoch = int(row.get("epoch", -1))
        if best_epoch < 0:
            raise RuntimeError(f"Metric '{key}' not found in metrics")
        return best_val, best_epoch
