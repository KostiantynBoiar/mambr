"""Mask and image array ops.

Pure numpy helpers on dense ``bool`` masks (shape ``(H, W)``) and images. Unit-tested against
hand-built masks. Implemented at M1.
"""

from __future__ import annotations

import numpy as np

from zoomtrack.geometry import Box


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    raise NotImplementedError("M1")


def area(mask: np.ndarray) -> int:
    """Number of set pixels in a boolean mask."""
    raise NotImplementedError("M1")


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of ``a`` contained in ``b``: ``|a & b| / |a|`` (0 if ``a`` is empty)."""
    raise NotImplementedError("M1")


def bbox(mask: np.ndarray) -> Box:
    """Tight bounding box ``(x0, y0, x1, y1)`` of a boolean mask."""
    raise NotImplementedError("M1")


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Logical OR of a list of same-shape boolean masks."""
    raise NotImplementedError("M1")


def downscale_nn(image: np.ndarray, factor: float) -> np.ndarray:
    """Nearest-neighbour downscale of an image/mask by ``factor``."""
    raise NotImplementedError("M1")


def upscale_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Upscale a boolean mask to ``(height, width)`` (nearest-neighbour)."""
    raise NotImplementedError("M1")


def crop_img(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``image`` to ``box`` ``(x0, y0, x1, y1)``."""
    raise NotImplementedError("M1")


def paste_tile(full_hw: tuple[int, int], tile_mask: np.ndarray, box: Box) -> np.ndarray:
    """Place a tile-local mask into a full-frame mask at ``box``.

    Args:
        full_hw: ``(H, W)`` of the full frame.
        tile_mask: Boolean mask in tile-local coordinates.
        box: The tile's box in full-frame coordinates.

    Returns:
        A full-frame boolean mask with ``tile_mask`` pasted in.
    """
    raise NotImplementedError("M1")
