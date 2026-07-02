"""Evaluation: bucket masklets small/large by area, then invoke the official ``veval``.

Reports pHOTA (with the pDetA/pAssA decomposition), cgF1, and TETA per subset. The headline number
is small-masklet pHOTA. The metric itself is computed by ``veval`` (facebookresearch/sam3) — we only
bucket and dispatch. Implemented at M0/M5.
"""

from __future__ import annotations

from pathlib import Path

from zoomtrack.config import Config
from zoomtrack.types import Masklet


class Evaluator:
    """Bucket predictions small/large and score each subset with veval."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (supplies the small-masklet area percentile + metrics).
        """
        self.config = config or Config()

    def bucket(self, masklets: list[Masklet]) -> dict[str, list[Masklet]]:
        """Split masklets into ``{"small": [...], "large": [...]}`` by mask-area percentile.

        Args:
            masklets: Masklets to bucket.

        Returns:
            The small/large partition.
        """
        raise NotImplementedError("M5: area-percentile bucketing")

    def score(self, pred_json: Path, gt_json: Path) -> dict[str, dict[str, float]]:
        """Invoke veval on predictions vs ground truth, per subset.

        Args:
            pred_json: Predicted VEval JSON (from :mod:`zoomtrack.writer`).
            gt_json: Ground-truth VEval JSON.

        Returns:
            ``{subset: {metric: value}}`` for small / large / overall.
        """
        raise NotImplementedError("M0/M5: dispatch to veval; never re-implement the metric")
