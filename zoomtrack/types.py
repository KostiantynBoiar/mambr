"""Shared record types for ZoomTrack.

A :class:`Detection` is one per-frame instance (a full-frame mask + score + provenance). A
:class:`Masklet` is one tracked object across time (``{frame_index -> Detection}``). Masks are dense
boolean arrays in development and unit tests; at scale they are stored/serialised as COCO-RLE, but
the in-memory contract everywhere in the pipeline is a full-frame ``bool`` array of shape ``(H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# provenance of a detection: coarse pass, a zoomed tile, or a fused instance.
Source = str  # one of {"coarse", "tile", "fused"}


@dataclass
class Detection:
    """A single per-frame instance in full-frame coordinates.

    Attributes:
        mask: Dense ``bool`` array of shape ``(H, W)`` (full frame).
        score: Confidence in ``[0, 1]``.
        source: Where it came from: ``"coarse"``, ``"tile"``, or ``"fused"``.
        tile_id: Index of the tile it was found in (``None`` for coarse/fused).
        track_id: Assigned during tracking (``None`` before linking).
    """

    mask: np.ndarray
    score: float
    source: Source
    tile_id: int | None = None
    track_id: int | None = None


@dataclass
class Masklet:
    """One tracked object across frames.

    Attributes:
        track_id: Stable identity across the clip.
        frames: Map from frame index to the :class:`Detection` at that frame (gaps allowed).
    """

    track_id: int
    frames: dict[int, Detection] = field(default_factory=dict)
