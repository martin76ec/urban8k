"""Unit tests for masked temporal pooling."""

from __future__ import annotations

import torch

from urban8k_ctransformer.models.ctransformer import CTransformerClassifier


def test_masked_pooling_ignores_padding() -> None:
    # Manually verify the static _masked_mean helper.
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 2:] = True  # sample 0: only first 2 frames valid
    pooled = CTransformerClassifier._masked_mean(x, mask)
    expected_0 = x[0, :2].mean(dim=0)
    expected_1 = x[1].mean(dim=0)
    assert torch.allclose(pooled[0], expected_0)
    assert torch.allclose(pooled[1], expected_1)


def test_masked_pooling_all_padding_does_not_nan() -> None:
    x = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)  # all padding
    pooled = CTransformerClassifier._masked_mean(x, mask)
    assert torch.isfinite(pooled).all()


def test_masked_pooling_no_mask_matches_mean() -> None:
    x = torch.randn(3, 5, 4)
    pooled = CTransformerClassifier._masked_mean(x, None)
    assert torch.allclose(pooled, x.mean(dim=1))


def test_changing_only_padded_frames_does_not_change_output() -> None:
    x = torch.randn(2, 6, 5)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 3:] = True
    mask[1, 4:] = True
    pooled_a = CTransformerClassifier._masked_mean(x, mask)
    x2 = x.clone()
    x2[0, 3:] = 1234.0  # modify only padded positions of sample 0
    x2[1, 4:] = -999.0
    pooled_b = CTransformerClassifier._masked_mean(x2, mask)
    assert torch.allclose(pooled_a, pooled_b)
