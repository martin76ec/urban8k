"""Unit tests for PositionalEncoding."""

from __future__ import annotations

import pytest
import torch

from urban8k_ctransformer.models.positional_encoding import PositionalEncoding


def test_pe_preserves_shape() -> None:
    pe = PositionalEncoding(d_model=32, max_len=128, dropout=0.0)
    x = torch.randn(4, 20, 32)
    out = pe(x)
    assert out.shape == x.shape


def test_pe_deterministic_in_eval() -> None:
    pe = PositionalEncoding(d_model=16, max_len=64, dropout=0.5)
    pe.eval()
    x = torch.randn(2, 10, 16)
    out1 = pe(x)
    out2 = pe(x)
    assert torch.allclose(out1, out2)


def test_pe_exceeding_max_len_raises() -> None:
    pe = PositionalEncoding(d_model=8, max_len=10, dropout=0.0)
    x = torch.randn(1, 11, 8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(x)


def test_pe_no_trainable_params() -> None:
    pe = PositionalEncoding(d_model=8, max_len=16, dropout=0.0)
    n_params = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    assert n_params == 0


def test_pe_max_len_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_len"):
        PositionalEncoding(d_model=8, max_len=0)
