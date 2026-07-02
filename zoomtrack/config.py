"""Central configuration for ZoomTrack.

Single source of truth for every constant (paths, tiling, selector, fusion, tracking, backbone,
eval). Defaults are overridable via a YAML file (``zoomtrack/configs/*.yaml``) or ``ZOOMTRACK_*``
environment variables. Mirrors the nested-dataclass ``Config`` pattern with a ``load()`` classmethod.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo root (this file is zoomtrack/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(env_var: str, default: Path) -> Path:
    """Return ``$env_var`` as an expanded path if set, else ``default``."""
    return Path(os.path.expanduser(os.environ.get(env_var, str(default))))


@dataclass
class PathsConfig:
    """Filesystem locations.

    Attributes:
        data_root: Where SA-FARI lives (``$ZOOMTRACK_DATA_ROOT``).
        cache_root: Part-A per-``(video, frame, tile)`` segmenter cache (``$ZOOMTRACK_CACHE_ROOT``).
        checkpoint_root: Trained tile-selector weights (``$ZOOMTRACK_CKPT_ROOT``).
    """

    data_root: Path = field(
        default_factory=lambda: _env_path("ZOOMTRACK_DATA_ROOT", _REPO_ROOT / "data")
    )
    cache_root: Path = field(
        default_factory=lambda: _env_path("ZOOMTRACK_CACHE_ROOT", _REPO_ROOT / "cache")
    )
    checkpoint_root: Path = field(
        default_factory=lambda: _env_path("ZOOMTRACK_CKPT_ROOT", _REPO_ROOT / "checkpoints")
    )


@dataclass
class DataConfig:
    """SA-FARI dataset facts (confirmed at M0; see SA-FARI_dataset_reference.md).

    Attributes:
        fps: Frame rate of the ``JPEGImages_6fps`` timeline the labels align to.
        frames_subdir: Media subdirectory holding the 6 fps frames.
    """

    fps: int = 6
    frames_subdir: str = "JPEGImages_6fps"


@dataclass
class TilingConfig:
    """Coarse pass + tiling geometry.

    Attributes:
        tile: Square tile side (px) in full-resolution coordinates.
        overlap: Overlap (px) between neighbouring tiles; edge tiles are clamped.
        downscale: Factor for the cheap coarse pass on the full frame (e.g. 0.25).
    """

    tile: int = 512
    overlap: int = 64
    downscale: float = 0.25


@dataclass
class SelectorConfig:
    """Tile-selection policy.

    Attributes:
        mode: ``"heuristic"`` (uncovered-area, default) or ``"learned"`` (trained head).
        tile_budget: Max tiles zoomed per frame (held FIXED across heuristic/learned comparisons).
        min_score: Heuristic threshold on the uncovered-area fraction.
        threshold: Learned-head probability threshold.
    """

    mode: str = "heuristic"
    tile_budget: int = 8
    min_score: float = 0.2
    threshold: float = 0.5


@dataclass
class FusionConfig:
    """Cross-scale fusion link rule.

    Attributes:
        iou_thresh: Link two masks if IoU >= this (same animal at two scales).
        cont_thresh: Link if containment(smaller, larger) >= this (fragment inside a bigger mask).
        mode: ``"union"`` (default, preserves recall) or ``"prefer_tile"`` (higher-res boundaries).
    """

    iou_thresh: float = 0.5
    cont_thresh: float = 0.8
    mode: str = "union"


@dataclass
class TrackingConfig:
    """Greedy-IoU temporal linking.

    Attributes:
        iou_thresh: Minimum IoU to associate an instance to a track across frames.
        max_gap: Frames a track may go unmatched before death (re-attachment window).
    """

    iou_thresh: float = 0.3
    max_gap: int = 5


@dataclass
class BackboneConfig:
    """Frozen segmentation backbone.

    Attributes:
        name: ``"sam3"`` (primary), ``"sam2"`` (fallback), or ``"mock"`` (tests).
        weights: Path/id of the frozen checkpoint (unused by the mock).
        min_area: Mock-only minimum mask area (px); reproduces the small-animal miss.
    """

    name: str = "mock"
    weights: str = ""
    min_area: int = 256


@dataclass
class EvalConfig:
    """Small/large bucketing + scoring.

    Attributes:
        small_percentile: Masklet-area percentile below which a masklet is "small".
        metrics: Metrics reported (computed by the official veval; never re-implemented).
    """

    small_percentile: float = 0.25
    metrics: tuple[str, ...] = ("pHOTA", "pDetA", "pAssA", "cgF1")


@dataclass
class Config:
    """Top-level configuration aggregating every section.

    Attributes:
        paths: Filesystem locations.
        data: SA-FARI dataset facts.
        tiling: Coarse pass + tiling geometry.
        selector: Tile-selection policy.
        fusion: Cross-scale fusion link rule.
        tracking: Temporal linking.
        backbone: Frozen backbone choice.
        eval: Small/large bucketing + metrics.
        seed: Random seed.
        raw: Raw parsed YAML, for any extra keys.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        """Build a config from defaults, then overlay a YAML file if given.

        Args:
            path: Optional YAML config path; defaults-only when ``None``.

        Returns:
            The resolved :class:`Config`.
        """
        cfg = cls()
        if path is not None:
            with open(path) as f:
                cfg.raw = yaml.safe_load(f) or {}
            cfg._apply(cfg.raw)
        return cfg

    def _apply(self, raw: dict[str, Any]) -> None:
        """Overlay parsed YAML onto the nested sections in place.

        Args:
            raw: Parsed YAML mapping; recognised sections mirror the dataclass field names, plus a
                top-level ``seed``.
        """
        sections = ("paths", "data", "tiling", "selector", "fusion", "tracking", "backbone", "eval")
        for section in sections:
            values = raw.get(section)
            if not values:
                continue
            sub = getattr(self, section)
            for key, value in values.items():
                if not hasattr(sub, key):
                    continue
                if section == "paths":
                    value = Path(os.path.expanduser(str(value)))
                setattr(sub, key, value)
        if "seed" in raw:
            self.seed = int(raw["seed"])
