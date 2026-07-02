"""Cross-scale fusion tests (M1) — the four hard cases + order-independence.

These guard the core novelty. On synthetic full-frame masks (no backbone needed):
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="M1: implement zoomtrack.fusion")


def test_coarse_and_tile_same_animal_merge() -> None:
    """Coarse + tile detections of one animal (high IoU) collapse to a single instance."""
    ...


def test_fragments_of_large_animal_stitched_by_coarse() -> None:
    """Tile fragments of one large animal are stitched via containment in the coarse mask."""
    ...


def test_small_animal_missed_by_coarse_is_added() -> None:
    """A tile-only detection the coarse pass missed is added as a new instance (the recall win)."""
    ...


def test_two_distinct_nearby_animals_stay_separate() -> None:
    """Two nearby but distinct animals (low IoU, low mutual containment) are not merged."""
    ...


def test_order_independence() -> None:
    """``fuse`` yields the same result under any permutation of the input detections."""
    ...
