"""Deterministic seeding across random / numpy / torch."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and (if importable) ``torch``.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
