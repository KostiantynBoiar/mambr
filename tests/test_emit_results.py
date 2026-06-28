"""Format-invariant gate for the S2 canary (no scorer, no GPU).

Emits the test split and checks that every line matches the eval_b1.py contract: 9 tab
fields, bracketed float vectors of length 11/19/5 in ActY/ActN/Spe order, the right number
of rows, and that the emitted labels round-trip back to the ClipRecord vectors.

Run: ``mammalps-env/bin/pytest -q tests/test_emit_results.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mammalps_b1.config import Config
from mammalps_b1.data.splits import SplitLoader
from mammalps_b1.scripts.emit_results import ResultsEmitter

_CONFIG = Config()
_DATA_PRESENT = (SplitLoader(_CONFIG).metadata_dir / "test.csv").exists()
pytestmark = pytest.mark.skipif(not _DATA_PRESENT, reason="benchmark_1 metadata CSVs not fetched")


def _parse_vec(field: str) -> np.ndarray:
    assert field.startswith("[") and field.endswith("]"), f"vector not bracketed: {field!r}"
    return np.fromstring(field[1:-1], dtype=float, sep=",")  # mirrors eval_b1.parse_line


def test_emit_matches_eval_contract(tmp_path: Path) -> None:
    """Emitted rows have the right shape, count, and round-tripping labels."""
    loader = SplitLoader(_CONFIG)
    records = loader.load("test")
    n_seg = _CONFIG.train.num_test_segments
    sizes = loader.labels.sizes  # {"ActY": 11, "ActN": 19, "Spe": 5}

    out = ResultsEmitter(_CONFIG, loader.labels, loader).emit("test", tmp_path / "r.txt")
    lines = out.read_text().splitlines()

    # one row per (clip x segment)
    assert len(lines) == len(records) * n_seg == 1244 * n_seg

    by_record = {r.sample_id: r for r in records}
    ids, sample_ids = set(), {}
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 9, f"expected 9 fields, got {len(fields)}"
        row_id, sample_id, seg = (int(fields[i]) for i in range(3))

        # preds and labels, both in ActY, ActN, Spe order
        order = ["ActY", "ActN", "Spe"]
        preds = {t: _parse_vec(fields[3 + i]) for i, t in enumerate(order)}
        labels = {t: _parse_vec(fields[6 + i]) for i, t in enumerate(order)}
        for t in order:
            assert preds[t].shape == (sizes[t],)
            assert labels[t].shape == (sizes[t],)

        # labels round-trip to the source ClipRecord
        rec = by_record[sample_id]
        assert np.argmax(labels["ActY"]) == np.argmax(rec.acty_onehot)
        assert np.argmax(labels["Spe"]) == np.argmax(rec.spe_onehot)
        assert np.array_equal(labels["ActN"].astype(int), rec.actn_multihot.astype(int))

        ids.add(row_id)
        sample_ids.setdefault(sample_id, 0)
        sample_ids[sample_id] += 1

    assert len(ids) == len(lines), "row ids must be unique"
    assert len(sample_ids) == 1244, "must aggregate to exactly 1244 clips"
    assert all(count == n_seg for count in sample_ids.values())
