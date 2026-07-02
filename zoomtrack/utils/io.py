"""Part-A cache IO: frozen-backbone detections keyed by ``(video, frame, tile)``.

The coarse and per-tile detections from the frozen backbone are computed once on a GPU and cached
to ``config.paths.cache_root`` so Part B (select / fuse / track / eval) can iterate cheaply without
re-running the backbone. Implemented at M4 (caching pass).
"""

from __future__ import annotations

from pathlib import Path

from zoomtrack.types import Detection


def cache_path(cache_root: Path, video: str, frame: int, tile: int | None) -> Path:
    """Return the on-disk path for one cache entry.

    Args:
        cache_root: Root cache directory (``config.paths.cache_root``).
        video: Video id.
        frame: Frame index.
        tile: Tile index, or ``None`` for the coarse (full-frame) pass.

    Returns:
        The path the entry is stored at.
    """
    raise NotImplementedError("M4: define the (video, frame, tile) cache layout")


def write_detections(path: Path, detections: list[Detection]) -> None:
    """Serialise detections (RLE masks) to the cache.

    Args:
        path: Destination from :func:`cache_path`.
        detections: Detections to persist.
    """
    raise NotImplementedError("M4: RLE-encode + write")


def read_detections(path: Path) -> list[Detection]:
    """Load cached detections (decoding RLE masks to dense arrays).

    Args:
        path: A path from :func:`cache_path`.

    Returns:
        The cached detections.
    """
    raise NotImplementedError("M4: read + RLE-decode")
