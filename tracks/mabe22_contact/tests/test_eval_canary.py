"""Format-invariant gate for the MABe22 eval contract (M1).

Confirms the leakage-safe split and that the scorer runs on dummy predictions — without
training anything. Auto-skips if the labels file has not been fetched.

Run from the repo root: ``mammalps-env/bin/pytest -q``
"""

from __future__ import annotations

import numpy as np
import pytest

from mabe22.config import Config
from mabe22.data.labels import MABeLabels
from mabe22.data.split import make_sequence_split
from mabe22.eval import MABeScorer

_CONFIG = Config()
_DATA_PRESENT = (_CONFIG.paths.data_root / _CONFIG.data.labels_file).exists()
pytestmark = pytest.mark.skipif(not _DATA_PRESENT, reason="MABe22 labels not fetched")


def test_split_is_leakage_safe() -> None:
    """Train/val/test sequences are disjoint and cover every sequence."""
    sids = MABeLabels(_CONFIG).sequence_ids()
    split = make_sequence_split(sids, _CONFIG)
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(sids)


def test_scorer_runs_on_dummy_preds() -> None:
    """The scorer returns per-behaviour + macro F1/AP for random preds over real labels."""
    labels = MABeLabels(_CONFIG)
    split = make_sequence_split(labels.sequence_ids(), _CONFIG)
    frames = labels.frames_for(split["test"][:8])  # a few sequences -> fast
    behaviours = _CONFIG.tasks.behaviours
    assert frames.labels.shape[1] == len(behaviours)

    preds = np.random.default_rng(0).random(frames.labels.shape)
    s = MABeScorer(_CONFIG).score(preds, frames.labels)
    for b in behaviours:
        assert set(s[b]) == {"f1", "ap", "n_pos", "n"}
    assert "f1" in s["macro"] and "ap" in s["macro"]
