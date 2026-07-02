"""SAM 3 adapter (primary backbone) — phrase-prompted, frozen, inference-only.

Prompted by the species phrase; exhaustive detect-segment-track. Weights are loaded frozen
(``eval()`` + ``requires_grad_(False)``). Implemented at the backbone milestone (M4).
"""

from __future__ import annotations

import numpy as np

from zoomtrack.config import Config
from zoomtrack.types import Detection


class Sam3Backbone:
    """Frozen SAM 3 wrapped behind the :class:`~zoomtrack.backbone.base.Backbone` interface."""

    def __init__(self, config: Config | None = None) -> None:
        """Load frozen SAM 3 weights.

        Args:
            config: Project config (``backbone.weights``, device).
        """
        self.config = config or Config()
        raise NotImplementedError("M4: load frozen SAM 3 (eval + requires_grad_(False))")

    def segment(self, image: np.ndarray, prompt: object) -> list[Detection]:
        """Segment ``image`` given a species phrase prompt.

        Args:
            image: Downscaled full frame or tile crop.
            prompt: Species phrase (str).

        Returns:
            Per-instance detections in the image's coordinates.
        """
        raise NotImplementedError("M4")
