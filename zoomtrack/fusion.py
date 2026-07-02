"""Cross-scale fusion — the core novel step (build + unit-test FIRST, M1).

Merge coarse + zoomed-tile detections into one instance per animal while keeping genuinely distinct
nearby animals separate. All masks are in full-frame coordinates (tile masks pasted in first).

Link rule -> connected components -> merge::

    link(a, b) is TRUE iff  IoU(a, b) >= iou_thresh
                        OR  containment(smaller, larger) >= cont_thresh
    instances = connected components of the graph over detections with those edges
    per component: fused_mask = OR(tile masks) if mode=="prefer_tile" and any tile mask
                                else OR(all masks)            # "union" default preserves recall
                   fused_score = max(scores)

Correct on the four hard cases (see tests/test_fusion.py): coarse+tile of one animal -> merge;
fragments of one large animal across tiles -> the coarse mask stitches them; a small animal missed
by coarse -> tile-only singleton is ADDED (the recall win); two distinct nearby animals -> stay
separate. Must be order-independent.
"""

from __future__ import annotations

from zoomtrack.types import Detection


def fuse(
    detections: list[Detection],
    iou_thresh: float,
    cont_thresh: float,
    mode: str = "union",
) -> list[Detection]:
    """Fuse coarse + tile detections into deduplicated per-frame instances.

    Args:
        detections: All detections for one frame, masks in full-frame coordinates.
        iou_thresh: Link two masks if IoU >= this.
        cont_thresh: Link if containment(smaller, larger) >= this.
        mode: ``"union"`` (default) or ``"prefer_tile"``.

    Returns:
        One fused :class:`Detection` (``source="fused"``) per connected component.
    """
    raise NotImplementedError("M1: link rule -> connected components -> merge")
