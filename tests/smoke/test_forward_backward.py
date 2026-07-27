"""Smoke test: forward + backward + optimizer step on synthetic data."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from urban8k_ctransformer.config import ModelConfig
from urban8k_ctransformer.models import CTransformerClassifier


@pytest.mark.smoke
def test_forward_backward_smoke() -> None:
    cfg = ModelConfig(
        cnn_channels=32,
        cnn_frequency_bins=16,
        input_dim=512,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        num_classes=2,
        max_len=256,
    )
    model = CTransformerClassifier(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    B, M = 4, 64
    T_values = [40, 55, 60, 48]
    T_max = max(T_values)
    features = torch.zeros(B, 1, M, T_max)
    mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, t in enumerate(T_values):
        features[i, 0, :, :t] = torch.randn(1, M, t)
        mask[i, :t] = False
    labels = torch.randint(0, cfg.num_classes, (B,))

    logits = model(features, src_key_padding_mask=mask)
    assert logits.shape == (B, cfg.num_classes)
    assert torch.isfinite(logits).all()

    loss = criterion(logits, labels)
    assert torch.isfinite(loss).item()

    loss.backward()
    at_least_one_grad = False
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            at_least_one_grad = True
            break
    assert at_least_one_grad, "No parameter received a finite gradient"

    optimizer.step()
    # New forward should still produce finite logits.
    logits2 = model(features, src_key_padding_mask=mask)
    assert torch.isfinite(logits2).all()
