"""Tiling geometry tests (M1) — coverage, overlap, edge-clamping, coordinate round-trip."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="M1: implement zoomtrack.geometry")


def test_tiles_cover_frame() -> None:
    """The union of all tiles covers the whole frame (edge tiles clamped to the border)."""
    ...


def test_tiles_overlap() -> None:
    """Neighbouring tiles overlap by the configured amount."""
    ...


def test_coord_round_trip() -> None:
    """``full_to_tile`` then ``tile_to_full`` returns the original point."""
    ...
