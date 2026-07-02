"""SAM 2 adapter (fallback backbone) — box/point-prompted, frozen, inference-only.

Prompted by SA-FARI's exhaustive first-frame boxes/points (or automatic mask generation). Used when
SAM 3 weights are unavailable; the method is backbone-agnostic and the claim is a controlled
base-vs-(base+ZoomTrack) gain. Weights are loaded frozen (``eval()`` + ``requires_grad_(False)``).
Implemented at the backbone milestone (M4).
"""

from __future__ import annotations

import numpy as np

from zoomtrack.config import Config
from zoomtrack.types import Detection


class Sam2Backbone:
    """Frozen SAM 2 wrapped behind the :class:`~zoomtrack.backbone.base.Backbone` interface."""

    def __init__(self, config: Config | None = None) -> None:
        """Load frozen SAM 2 weights.

        Args:
            config: Project config (``backbone.weights``, device).
        """
        self.config = config or Config()
        raise NotImplementedError("M4: load frozen SAM 2 (eval + requires_grad_(False))")

    def segment(self, image: np.ndarray, prompt: object) -> list[Detection]:
        """Segment ``image`` given box/point prompts.

        Args:
            image: Downscaled full frame or tile crop.
            prompt: Boxes/points (or ``None`` for automatic mask generation).

        Returns:
            Per-instance detections in the image's coordinates.
        """
        raise NotImplementedError("M4")
