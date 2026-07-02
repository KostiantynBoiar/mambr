"""Train the learned tile-selector (the contribution) — M3.

Turns cached coarse detections + ground-truth masklets into per-tile labels, then fits the head::

    label(tile) = 1 iff (a GT animal's mask overlaps this tile)
                       AND (that GT animal is NOT matched by any coarse detection at IoU >= tau)

i.e. "the coarse pass missed an animal here". Positives are rare -> class-balanced BCE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zoomtrack.config import Config
from zoomtrack.geometry import Box
from zoomtrack.types import Detection, Masklet


def build_tile_labels(
    tiles: list[Box],
    coarse: list[Detection],
    gt: list[Masklet],
    frame: int,
    iou_match: float,
) -> np.ndarray:
    """Per-tile binary labels for one frame ("coarse missed an animal in this tile").

    Args:
        tiles: Candidate tiles for the frame.
        coarse: Coarse-pass detections for the frame.
        gt: Ground-truth masklets.
        frame: Frame index.
        iou_match: IoU threshold at which a coarse det "covers" a GT animal.

    Returns:
        ``int8`` array of shape ``(len(tiles),)`` with values in ``{0, 1}``.
    """
    raise NotImplementedError("M3: label = GT-in-tile AND not-covered-by-coarse")


def train(config: Config, out_path: Path) -> None:
    """Fit the selector head on cached detections + GT and save it.

    Args:
        config: Project config (paths, selector).
        out_path: Where to write the trained head.
    """
    raise NotImplementedError("M3: features -> head -> class-balanced BCE -> save")
