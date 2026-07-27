# urban8k-ctransformer

CNN backbone + Transformer encoder for UrbanSound8K audio classification. The LSTM from a previous CRNN project is replaced by a `torch.nn.TransformerEncoder`; the CNN backbone is preserved, and a masked mean-pooling head produces logits over the 10 UrbanSound8K classes.

## Architecture

```text
log-Mel spectrogram          (B, 1, 128, T)
CNN backbone                 (B, 32, 16, T)
reorder temporal             (B, T, 512)    # 32 × 16
input projection             (B, T, 256)
positional encoding          (B, T, 256)
TransformerEncoder (2 layers)(B, T, 256)
masked mean pooling          (B, 256)
linear head                  (B, 10)
```

`T` is variable; padded frames are ignored via `src_key_padding_mask` and excluded from the mean pool.

## Results

Run: `ctransformer_seed42` (seed 42, NVIDIA H200, CUDA 12.8, torch 2.9.1+cu128).

Trained 18 epochs (cancelled manually; best val macro-F1 at epoch 13). Evaluated on the held-out test fold (fold 10, 837 samples):

| Metric        | Value  |
|---------------|--------|
| Test accuracy | 0.749  |
| Test macro-F1 | 0.763  |

Per-class F1 (selected):

| Class              | F1    |
|--------------------|-------|
| gun_shot           | 0.952 |
| jackhammer         | 0.904 |
| car_horn           | 0.857 |
| drilling           | 0.842 |
| street_music       | 0.771 |
| dog_bark           | 0.731 |
| engine_idling      | 0.714 |
| air_conditioner    | 0.714 |
| children_playing   | 0.659 |
| siren              | 0.481 |

Full results and logs:

- [`summary.json`](artifacts/runs/ctransformer_seed42/summary.json) — test metrics + per-class classification report
- [`metrics.jsonl`](artifacts/runs/ctransformer_seed42/metrics.jsonl) — one line per epoch (train/val loss, accuracy, macro-F1, LR, time, VRAM)
- [`run.log`](artifacts/runs/ctransformer_seed42/run.log) — full training + evaluation stdout/stderr
- [`config.yaml`](artifacts/runs/ctransformer_seed42/config.yaml) — resolved configuration
- [`metadata.json`](artifacts/runs/ctransformer_seed42/metadata.json) — environment (host, torch, CUDA, GPU, seed)
- [`figures/learning_curves.png`](artifacts/runs/ctransformer_seed42/figures/learning_curves.png) — loss + val macro-F1 per epoch
- [`figures/confusion_matrix.png`](artifacts/runs/ctransformer_seed42/figures/confusion_matrix.png) — 10×10 confusion matrix
- [`figures/confusion_matrix.csv`](artifacts/runs/ctransformer_seed42/figures/confusion_matrix.csv) — raw confusion matrix

## Splits

Following the official UrbanSound8K folds:

| Split | Folds |
|-------|-------|
| train | 1–8   |
| val   | 9     |
| test  | 10    |

The test fold is never touched until `evaluate.py`.

## Commands

```bash
make setup                                                 # install deps
make all                                                   # lint + typecheck + tests (no GPU/data needed)
make data   CONFIG=configs/h200.yaml                       # extract WAVs + compute normalization stats
make train  CONFIG=configs/h200.yaml RUN_ID=ctransformer_seed42
make evaluate CONFIG=configs/h200.yaml RUN_ID=ctransformer_seed42
```

`make train` never downloads data implicitly. Each run is self-contained under `artifacts/runs/<run_id>/` with config, metadata, per-epoch metrics, checkpoints, figures, and a full `run.log`. Runs cannot be overwritten accidentally; use `--resume` to continue.

## Dataset

Place UrbanSound8K under `data/raw/UrbanSound8K/`:

```text
data/raw/UrbanSound8K/
├── UrbanSound8K.csv
└── audio/
    ├── fold1/ ...
    └── fold10/ ...
```

Download from HuggingFace (`danavery/urbansound8K` — audio is stored inside parquet shards, `prepare_data.py` extracts them automatically):

```bash
huggingface-cli download danavery/urbansound8K --repo-type dataset --local-dir data/raw/UrbanSound8K
```

`data/` and `artifacts/` are git-ignored (except committed results referenced above).