"""Temporal linking — associate per-frame fused instances into masklets.

Greedy IoU association with birth/death and a small re-attachment gap. Deliberately simple; the
interface is stable so a stronger associator can be swapped in later. Implemented at M1.
"""

from __future__ import annotations

from zoomtrack.types import Detection, Masklet


def link_tracks(frames: list[list[Detection]], iou_thresh: float, max_gap: int) -> list[Masklet]:
    """Link per-frame fused detections into masklets.

    Args:
        frames: Per-frame lists of fused detections, in frame order.
        iou_thresh: Minimum IoU to associate a detection to an existing track.
        max_gap: Frames a track may go unmatched before it dies (re-attachment window).

    Returns:
        The resulting masklets, each with a stable ``track_id``.
    """
    raise NotImplementedError("M1: greedy-IoU association with birth/death + gap")
