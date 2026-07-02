"""Mock backbone — connected components with a minimum-area floor.

Stands in for a real segmenter in unit tests and the synthetic smoke run, and deliberately
reproduces the small-animal failure: a blob below ``config.backbone.min_area`` in the downscaled
full frame is dropped, but the same blob passes the floor inside a full-resolution tile crop.
Implemented at M1.
"""

from __future__ import annotations

import numpy as np

from zoomtrack.config import Config
from zoomtrack.types import Detection


class MockBackbone:
    """Connected-components segmenter with a min-area floor (no learning, no weights)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``backbone.min_area`` is the floor).
        """
        self.config = config or Config()

    def segment(self, image: np.ndarray, prompt: object = None) -> list[Detection]:
        """Return one detection per connected component above the min-area floor.

        Args:
            image: A binary/greyscale image (blobs are foreground). Coarse frames are downscaled, so
                small blobs fall below the floor and are dropped — the small-animal miss.
            prompt: Ignored.

        Returns:
            Detections (``source="coarse"``; caller relabels tile detections).
        """
        raise NotImplementedError("M1: connectedComponents + min-area floor")
