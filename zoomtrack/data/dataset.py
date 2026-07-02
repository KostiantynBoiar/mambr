"""SA-FARI loader to the ``_ext`` schema (M0).

Loads the annotation JSONs (five top-level fields) and the 6 fps frames: per-video, per-frame RLE
masklets frame-aligned to ``file_names``, plus taxonomy and ``location_id``. Negatives are
``video_np_pairs`` with ``num_masklets == 0`` (must yield nothing; a false positive there is
penalised by cgF1). RLE is decoded with ``pycocotools``. Exact fields confirmed at M0 — see
``SA-FARI_dataset_reference.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zoomtrack.config import Config
from zoomtrack.types import Masklet


@dataclass
class Video:
    """One SA-FARI video sample.

    Attributes:
        video_id: Identifier.
        file_names: Ordered 6 fps frame paths.
        gt_masklets: Ground-truth masklets (empty for a negative pair).
        taxonomy: Species/taxon label.
        location_id: Camera/location identifier (for cross-location splits).
    """

    video_id: str
    file_names: list[str]
    gt_masklets: list[Masklet]
    taxonomy: str
    location_id: str


class SAFARI:
    """Iterate SA-FARI videos + ground-truth masklets."""

    def __init__(self, split: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            split: Dataset split (e.g. ``"val"``).
            config: Project config (``paths.data_root``, ``data.fps``).
        """
        self.split = split
        self.config = config or Config()
        raise NotImplementedError("M0: parse the _ext JSONs + pycocotools RLE")

    def videos(self) -> list[Video]:
        """Return all videos in the split."""
        raise NotImplementedError("M0")

    def frame(self, video: str, index: int) -> np.ndarray:
        """Load one decoded frame ``(H, W, 3)``.

        Args:
            video: Video id.
            index: Frame index into ``file_names``.

        Returns:
            The decoded RGB frame.
        """
        raise NotImplementedError("M0")
