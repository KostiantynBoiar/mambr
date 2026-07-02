"""Tile selection — which tiles to zoom (heuristic default, learned contribution).

Both selectors share ``select(tiles, coarse_masks, image_hw) -> [(tile, score)]``, ranked
descending; the pipeline zooms the top ``tile_budget``. The heuristic (uncovered-area) is real and
the default; the learned head (``zoomtrack.selector``) is the contribution, trained at M3.
Implemented (heuristic) at M1.
"""

from __future__ import annotations

import numpy as np

from zoomtrack.config import Config
from zoomtrack.geometry import Box


class HeuristicSelector:
    """Score tiles by uncovered area — missed animals live where coarse found nothing."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()

    def select(
        self, tiles: list[Box], coarse_masks: list[np.ndarray], image_hw: tuple[int, int]
    ) -> list[tuple[Box, float]]:
        """Rank tiles by ``1 - covered_fraction`` (descending).

        Args:
            tiles: Candidate tiles from :func:`zoomtrack.geometry.make_tiles`.
            coarse_masks: Coarse-pass full-frame masks.
            image_hw: ``(H, W)`` of the frame.

        Returns:
            ``[(tile, score)]`` sorted by score descending.
        """
        raise NotImplementedError("M1: uncovered-area score = 1 - covered/tile_area")


class LearnedSelector:
    """Predict P(a missed animal is in this tile) with a trained head (M3)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (supplies the head checkpoint + threshold).
        """
        self.config = config or Config()

    def select(
        self, tiles: list[Box], coarse_masks: list[np.ndarray], image_hw: tuple[int, int]
    ) -> list[tuple[Box, float]]:
        """Rank tiles by the head's predicted miss-probability (descending).

        Args:
            tiles: Candidate tiles.
            coarse_masks: Coarse-pass full-frame masks.
            image_hw: ``(H, W)`` of the frame.

        Returns:
            ``[(tile, score)]`` sorted by score descending.
        """
        raise NotImplementedError("M3: features -> head -> probability")
