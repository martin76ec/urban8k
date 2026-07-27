"""Device resolution."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string, falling back to CPU if CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
