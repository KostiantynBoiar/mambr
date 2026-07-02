"""Per-tile features for the learned selector (M3).

Features = pooled frozen-encoder features over the tile, concatenated with coarse statistics in the
tile (uncovered fraction, number of low-confidence coarse dets, max coarse score). These are cached
alongside the Part-A detections so training/inference never re-runs the backbone.
"""

from __future__ import annotations

import numpy as np

from zoomtrack.geometry import Box
from zoomtrack.types import Detection


def tile_features(
    tile: Box, coarse: list[Detection], encoder_feats: np.ndarray | None = None
) -> np.ndarray:
    """Build the feature vector for one tile.

    Args:
        tile: The tile box.
        coarse: Coarse-pass detections for the frame.
        encoder_feats: Optional pooled encoder features over the tile (cached).

    Returns:
        A 1-D ``float32`` feature vector.
    """
    raise NotImplementedError("M3: pooled encoder feats + coarse stats in the tile")
