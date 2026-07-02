"""Serialise predicted masklets to SA-Co/VEval JSON (RLE) for the official ``veval`` scorer.

Emitting the scorer's native format means the metric is never re-implemented — ``veval`` reads this
directly. Implemented at M0/M5.
"""

from __future__ import annotations

from pathlib import Path

from zoomtrack.types import Masklet


def write_veval_json(masklets: list[Masklet], video: str, out_path: Path) -> None:
    """Write predicted masklets as SA-Co/VEval JSON (RLE-encoded masks).

    Args:
        masklets: Predicted masklets for one video.
        video: Video id (for the output record).
        out_path: Destination JSON path.
    """
    raise NotImplementedError("M0/M5: RLE-encode masks + emit the VEval schema")
