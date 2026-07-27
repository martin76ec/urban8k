"""Prepare data: extract WAVs from HF parquet shards and compute normalization stats.

The HuggingFace ``danavery/urbansound8K`` dataset stores audio inside parquet
shards (an ``audio`` column with embedded WAV bytes), not as loose ``.wav``
files. This script:

1. Reads ``UrbanSound8K.csv`` for metadata.
2. Scans the parquet shards under ``data/raw/UrbanSound8K/data/`` and writes
   each row's audio bytes to ``data/raw/UrbanSound8K/audio/fold{N}/{name}``
   so the rest of the pipeline can read plain WAVs.
3. Computes per-frequency-bin log-Mel mean/std from the training folds only
   and saves them to ``data/processed/norm_stats.pkl``.
4. Writes ``train.csv`` / ``val.csv`` / ``test.csv`` splits.

If the WAV files already exist, step 2 is skipped (idempotent).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from urban8k_ctransformer.config import load_config
from urban8k_ctransformer.data import UrbanSound8KDataset, load_metadata, official_folds_split
from urban8k_ctransformer.utils.logging import get_logger

_logger = get_logger("urban8k.prepare")


def _extract_wavs_from_parquet(root: Path) -> None:
    """Read parquet shards and write WAVs to ``root/audio/fold{N}/``."""
    parquet_dir = root / "data"
    audio_dir = root / "audio"
    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"Parquet shards not found at {parquet_dir}. Download the dataset first:\n"
            "  huggingface-cli download danavery/urbansound8K --repo-type dataset "
            f"--local-dir {root}"
        )

    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet shards found in {parquet_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent: skip if all folds already populated.
    existing = list(audio_dir.glob("fold*/*.wav"))
    if len(existing) >= 8732:
        _logger.info("Found %d WAVs already extracted, skipping parquet extraction", len(existing))
        return

    _logger.info("Extracting WAVs from %d parquet shards -> %s", len(shards), audio_dir)
    n_written = 0
    for shard in tqdm(shards, desc="shards"):
        df = pd.read_parquet(shard, columns=["audio", "slice_file_name", "fold"])
        for row in df.itertuples(index=False):
            audio_bytes: bytes = row.audio["bytes"]
            fold = int(row.fold)
            name = str(row.slice_file_name)
            out_dir = audio_dir / f"fold{fold}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            if not out_path.exists():
                out_path.write_bytes(audio_bytes)
                n_written += 1
    _logger.info("Wrote %d new WAV files", n_written)


def _load_norm_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if not path.exists():
        return None, None
    with path.open("rb") as f:
        d = pickle.load(f)  # noqa: S301
    return d["mean"], d["std"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare UrbanSound8K metadata and stats")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.data.root)
    _logger.info("Loading metadata from %s", root)
    metadata = load_metadata(root)
    split = official_folds_split(metadata)

    # Step 1: extract WAVs from parquet shards if needed.
    _extract_wavs_from_parquet(root)

    # Step 2: save split metadata as CSVs for inspection.
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(out_dir / "train.csv", index=False)
    split.val.to_csv(out_dir / "val.csv", index=False)
    split.test.to_csv(out_dir / "test.csv", index=False)

    # Step 3: compute normalization stats from train fold only.
    _logger.info("Computing log-Mel stats over %d training samples", len(split.train))
    ds = UrbanSound8KDataset(split.train, root, cfg.data)
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