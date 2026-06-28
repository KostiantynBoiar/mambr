"""Deterministic seeding so we can report mean ± std over seeds (CLAUDE.md §3)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU + MPS/CUDA) for reproducible runs.

    Args:
        seed: The random seed.
        deterministic: If ``True``, also set cuDNN to deterministic mode.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        # torch not installed yet (e.g. during early env setup) — numpy/random still seeded.
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
