"""Leakage-safe split of MABe22 sequences (M1).

Split by **sequence** so every frame of a clip stays in one split — adjacent frames are
near-identical, so a random per-frame split would leak. Fixed and seeded.
"""

from __future__ import annotations

import numpy as np

from mabe22.config import Config


def make_sequence_split(seq_ids: list[str], config: Config | None = None) -> dict[str, list[str]]:
    """Partition sequence ids into train / val / test by sequence.

    Args:
        seq_ids: All labelled sequence ids.
        config: Project config (``eval.val_frac`` / ``eval.test_frac`` / ``seed``).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with **disjoint** sequences.
    """
    cfg = config or Config()
    ids = list(seq_ids)
    np.random.default_rng(cfg.seed).shuffle(ids)
    n = len(ids)
    n_test = round(n * cfg.eval.test_frac)
    n_val = round(n * cfg.eval.val_frac)
    return {
        "test": ids[:n_test],
        "val": ids[n_test : n_test + n_val],
        "train": ids[n_test + n_val :],
    }
