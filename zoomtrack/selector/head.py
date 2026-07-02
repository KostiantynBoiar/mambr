"""The small trainable selector head (M3).

A compact MLP/CNN predicting P(a missed animal is in this tile) from :mod:`zoomtrack.selector.features`.
The only trained component in the project (~10^4-10^5 params); held at identical capacity across the
heuristic-vs-learned comparison so a gain reflects information, not parameters. Class-balanced BCE
(positives are rare).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SelectorHead:
    """Small MLP/CNN head: tile features -> miss-probability."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        """Initialize.

        Args:
            in_dim: Feature dimension from :func:`zoomtrack.selector.features.tile_features`.
            hidden: Hidden width.
        """
        self.in_dim = in_dim
        self.hidden = hidden
        raise NotImplementedError("M3: build the head")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict miss-probabilities for a batch of tile features.

        Args:
            features: ``(N, in_dim)`` features.

        Returns:
            ``(N,)`` probabilities in ``[0, 1]``.
        """
        raise NotImplementedError("M3")

    def save(self, path: Path) -> None:
        """Persist the trained head."""
        raise NotImplementedError("M3")

    @classmethod
    def load(cls, path: Path) -> SelectorHead:
        """Load a trained head."""
        raise NotImplementedError("M3")
