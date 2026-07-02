"""Tile-selector tests — heuristic ranking now (M1); learned head shape later (M3)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="M1: implement HeuristicSelector; M3: LearnedSelector")


def test_heuristic_ranks_uncovered_tiles_first() -> None:
    """Tiles with more coarse-uncovered area rank above well-covered tiles."""
    ...
