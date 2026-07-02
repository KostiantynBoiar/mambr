"""Mask op tests (M1) — iou / area / containment / bbox / union / scale / crop / paste."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="M1: implement zoomtrack.masks")


def test_iou_and_containment() -> None:
    """IoU and containment match hand-computed values on overlapping masks."""
    ...


def test_paste_tile_places_mask() -> None:
    """``paste_tile`` puts a tile-local mask at the right full-frame location."""
    ...
