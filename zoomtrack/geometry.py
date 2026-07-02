"""Tiling geometry: cover a frame with overlapping tiles and map coordinates between scales.

Pure functions (no state, no IO). Unit-tested for coverage (the tile union covers the whole frame),
overlap, edge clamping, and coordinate round-trip. Implemented at M1.

A ``Box`` is ``(x0, y0, x1, y1)`` in full-frame pixel coordinates (half-open, ``x1``/``y1`` exclusive).
"""

from __future__ import annotations

Box = tuple[int, int, int, int]


def make_tiles(height: int, width: int, tile: int, overlap: int) -> list[Box]:
    """Cover a ``height x width`` frame with overlapping square tiles.

    Tiles step by ``tile - overlap``; the last row/column is clamped to the border so the union of
    all tiles always covers the frame exactly.

    Args:
        height: Frame height (px).
        width: Frame width (px).
        tile: Tile side (px).
        overlap: Overlap between neighbours (px).

    Returns:
        Tile boxes ``(x0, y0, x1, y1)`` in reading order.
    """
    raise NotImplementedError("M1: overlapping, edge-clamped tiling with full coverage")


def tile_to_full(point: tuple[int, int], tile_box: Box) -> tuple[int, int]:
    """Map a tile-local ``(x, y)`` to full-frame coordinates.

    Args:
        point: ``(x, y)`` in tile-local coordinates.
        tile_box: The tile's box in full-frame coordinates.

    Returns:
        ``(x, y)`` in full-frame coordinates.
    """
    raise NotImplementedError("M1")


def full_to_tile(point: tuple[int, int], tile_box: Box) -> tuple[int, int]:
    """Map a full-frame ``(x, y)`` to tile-local coordinates.

    Args:
        point: ``(x, y)`` in full-frame coordinates.
        tile_box: The tile's box in full-frame coordinates.

    Returns:
        ``(x, y)`` in tile-local coordinates.
    """
    raise NotImplementedError("M1")
