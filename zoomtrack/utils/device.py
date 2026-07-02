"""Torch device selection (cuda > mps > cpu)."""

from __future__ import annotations

import torch


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device.

    Args:
        pref: ``"auto"`` picks cuda > mps > cpu by availability; any other string
            (``"cpu"``, ``"mps"``, ``"cuda"``) is passed through unchanged.

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
