"""Unit tests for CTransformerClassifier output shapes."""

from __future__ import annotations

import pytest
import torch

from urban8k_ctransformer.config import ModelConfig
from urban8k_ctransformer.models import CTransformerClassifier


def _cfg(**overrides: object) -> ModelConfig:
    defaults = dict(
        cnn_channels=32,
        cnn_frequency_bins=16,
        input_dim=512,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        num_classes=10,
        max_len=512,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize("T", [50, 137])
def test_logits_shape_for_variable_T(T: int) -> None:
    cfg = _cfg()
    model = CTransformerClassifier(cfg)
    model.eval()
    B, M = 3, 128
    x = torch.randn(B, 1, M, T)
    logits = model(x)
    assert logits.shape == (B, cfg.num_classes)


def test_logits_shape_with_padding_mask() -> None:
    cfg = _cfg()
    model = CTransformerClassifier(cfg)
    model.eval()
    B, M, T = 4, 128, 64
    x = torch.randn(B, 1, M, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 40:] = True
    mask[2, 50:] = True
    logits = model(x, src_key_padding_mask=mask)
    assert logits.shape == (B, cfg.num_classes)
    assert torch.isfinite(logits).all()


def test_invalid_d_model_nhead_raises() -> None:
    with pytest.raises(ValueError, match="d_model.*nhead"):
        _cfg(d_model=33, nhead=4).validate()


def test_invalid_num_classes_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        _cfg(num_classes=1).validate()
