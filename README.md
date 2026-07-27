# urban8k-ctransformer

CNN backbone + Transformer encoder for UrbanSound8K classification. The LSTM of
the previous CRNN project is replaced by a `torch.nn.TransformerEncoder`; the
CNN backbone is preserved and a masked mean-pooling head produces logits over
the 10 UrbanSound8K classes.

## Tensor contract

```text
log-Mel spectrogram          (B, 1, M, T)
CNN backbone                 (B, 32, 16, T)
reorder temporal             (B, T, 512)   # 32 × 16
input projection             (B, T, 256)
positional encoding          (B, T, 256)
TransformerEncoder (2 layers)(B, T, 256)
masked mean pooling          (B, 256)
linear head                  (B, 10)
```

`T` is **not** fixed: the model accepts variable-length sequences. When a batch
contains padding, the model receives a `src_key_padding_mask` (True = padding)
and the mean pooling ignores padded frames.

## Repository layout

```text
urban8k-ctransformer/
├── pyproject.toml          # dependencies, ruff, mypy, pytest config
├── Makefile                # primary command surface
├── README.md
├── configs/
│   ├── base.yaml           # CPU-friendly defaults
│   └── h200.yaml           # NVIDIA H200 setup
├── scripts/
│   ├── prepare_data.py     # build metadata + train-set normalization stats
│   ├── train.py            # train one run
│   └── evaluate.py         # evaluate a run on the test fold
├── src/urban8k_ctransformer/
│   ├── config.py           # typed YAML configuration
│   ├── data/               # dataset, features, collate, splits
│   ├── models/             # CNN backbone, positional encoding, classifier
│   ├── training/           # engine, metrics, checkpointing (RunManager)
│   └── utils/              # seed, device, logging
├── tests/
│   ├── unit/               # positional encoding, shapes, masked pooling
│   └── smoke/              # forward/backward, end-to-end pipeline
├── data/                   # NOT in git (raw + processed)
└── artifacts/runs/         # NOT in git (one folder per run)
```

## Install

```bash
make setup    # uv sync --group dev
```

PyTorch is installed via `uv sync` using the project's `pyproject.toml`. On the
H200 server, verify the CUDA driver first and install a CUDA-compatible wheel
set if the default does not match:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Record these outputs in the run's `metadata.json` automatically.

## Dataset

Place UrbanSound8K under `data/raw/UrbanSound8K/` so that:

```text
data/raw/UrbanSound8K/
├── UrbanSound8K.csv
└── audio/
    ├── fold1/...
    └── fold10/...
```

`data/` and `artifacts/` are git-ignored.

## Splits

Following the official folds:

| Split | Folds     |
|-------|-----------|
| train | 1–8       |
| val   | 9         |
| test  | 10        |

The test fold is **never** touched until `evaluate.py`.

## Commands (Makefile)

```bash
make all                                            # lint + typecheck + unit + smoke (no GPU, no data)
make data   CONFIG=configs/h200.yaml                # build metadata + normalization stats
make train  CONFIG=configs/h200.yaml                # auto run-id
make train  CONFIG=configs/h200.yaml RUN_ID=ctransformer_seed42
make evaluate CONFIG=configs/h200.yaml RUN_ID=ctransformer_seed42
```

`make train` never prepares or downloads data implicitly. `make all` runs on a
laptop without GPU or dataset.

## Run artifacts

Each training run produces a self-contained folder:

```text
artifacts/runs/<run_id>/
├── config.yaml             # resolved configuration
├── metadata.json           # UTC date, hostname, git sha, Python, Torch, CUDA, GPU, seed
├── metrics.jsonl           # one JSON line per epoch
├── summary.json            # best epoch + final test metrics
├── checkpoints/
│   ├── best.pt             # best val_macro_f1
│   └── last.pt             # last epoch (for --resume)
└── figures/
    ├── learning_curves.png
    └── confusion_matrix.png (+ .csv)
```

A run **cannot be overwritten** by accident: re-running with the same `RUN_ID`
fails unless `--resume` is passed. `--resume <run_id>` loads `last.pt` and
continues logging into the same run folder.

## Configuration

All hyperparameters live in `configs/*.yaml` and are loaded into typed
dataclasses in `src/urban8k_ctransformer/config.py`. Validation at load time
enforces `d_model % nhead == 0`, `num_classes > 1`, and the CNN contract
`cnn_channels * cnn_frequency_bins == input_dim`.

Key fields (`configs/h200.yaml`):

| Section | Field              | Value |
|---------|--------------------|-------|
| data    | sample_rate        | 22050 |
| data    | n_mels             | 128   |
| data    | max_seconds        | 4.0   |
| model   | d_model / nhead    | 256 / 8 |
| model   | num_layers         | 2     |
| model   | dim_feedforward    | 512   |
| model   | dropout            | 0.1   |
| model   | num_classes        | 10    |
| model   | max_len            | 2048  |
| train   | batch_size         | 128   |
| train   | epochs             | 50    |
| train   | learning_rate      | 3e-4  |
| train   | weight_decay       | 1e-4  |
| train   | amp                | true  |
| train   | gradient_clip_norm | 1.0   |
| train   | early_stopping_patience | 10 |
| train   | selection_metric   | val_macro_f1 |

## Testing

```bash
make test     # unit tests: shapes, positional encoding, masked pooling
make smoke    # CPU smoke tests with synthetic tensors
```

Smoke tests do not require UrbanSound8K or a GPU and run in under a minute.

## Architecture notes

- The TransformerEncoder is the correct choice because this task **encodes and
  classifies an existing sequence**; it does not generate audio or tokens
  autoregressively.
- Self-attention has no implicit temporal order. The sinusoidal positional
  encoding gives the encoder the position of each frame so it can distinguish
  an event at the start of a clip from one at the end.
- Masked mean pooling summarizes all valid frames into a fixed-size vector
  before the linear head, ignoring padded positions.
- The CNN backbone is preserved from the previous CRNN project; its output
  contract `(B, 32, 16, T)` is checked at forward time against `input_dim=512`.

## Reproducibility checklist

- [x] Model contains a CNN and `nn.TransformerEncoder`; no LSTM/GRU.
- [x] Path `(B, 32, 16, T) → (B, T, 512) → (B, T, 256)` is tested.
- [x] Positional encoding + `src_key_padding_mask` for variable lengths.
- [x] Masked temporal pooling; final output `(B, 10)`.
- [x] Split, seed, optimizer, epochs, device and config are recorded.
- [x] Test reports accuracy, macro-F1, confusion matrix and per-class report.
- [x] Each run is self-contained with config, metadata, metrics, checkpoints, figures.
- [x] Existing runs cannot be overwritten; explicit `--resume` required.
- [x] `make all` passes without GPU or dataset.