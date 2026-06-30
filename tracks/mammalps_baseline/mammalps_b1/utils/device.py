"""Pick a compute device so the same code runs on the M4 (MPS) and on Kaggle (CUDA)."""

from __future__ import annotations

import torch


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device.

    Args:
        pref: ``"auto"`` to choose mps > cuda > cpu by availability, or an explicit
            device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns:
        The selected :class:`torch.device`.
    """
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
