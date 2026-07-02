"""The frozen-segmenter interface every backbone adapter implements.

One method, resolution-agnostic: it is called on the downscaled full frame (coarse pass) and on
full-resolution tile crops (zoom pass) identically. Adapters wrap a FROZEN, inference-only model —
never trained, no gradients.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from zoomtrack.types import Detection


class Backbone(Protocol):
    """A promptable segmenter returning per-instance detections for one image."""

    def segment(self, image: np.ndarray, prompt: object) -> list[Detection]:
        """Segment ``image`` under ``prompt`` into instance detections.

        Args:
            image: ``(H, W, 3)`` (or ``(H, W)``) image array — a downscaled full frame or a tile crop.
            prompt: Backbone-specific prompt (species phrase for SAM 3; boxes/points for SAM 2;
                ignored by the mock).

        Returns:
            One :class:`Detection` per instance, masks in the image's own coordinates.
        """
        ...
