"""Orchestration: coarse -> select -> zoom -> fuse -> track (+ a coarse-only baseline path).

Config-driven. The mock-backbone path is exercised by the M1 synthetic smoke test; the
real-backbone Part-A caching path is wired at M4.
"""

from __future__ import annotations

from zoomtrack.backbone.base import Backbone
from zoomtrack.config import Config
from zoomtrack.types import Masklet


class Pipeline:
    """Run ZoomTrack (or the coarse-only baseline) over a video."""

    def __init__(self, backbone: Backbone, config: Config | None = None) -> None:
        """Initialize.

        Args:
            backbone: A frozen segmenter (``sam3`` / ``sam2`` / ``mock``).
            config: Project config (tiling, selector, fusion, tracking).
        """
        self.backbone = backbone
        self.config = config or Config()

    def run(self, video: str) -> list[Masklet]:
        """Full pipeline: coarse -> select -> zoom -> fuse -> track.

        Args:
            video: Video id (frames resolved via ``config``).

        Returns:
            The predicted masklets.
        """
        raise NotImplementedError("M1 (mock path) / M4 (real backbone + cache)")

    def run_coarse_only(self, video: str) -> list[Masklet]:
        """Baseline path: coarse pass -> track, with no tiling/fusion.

        Args:
            video: Video id.

        Returns:
            The baseline masklets (for the M2 ablation).
        """
        raise NotImplementedError("M1 (mock path) / M4 (real backbone)")
