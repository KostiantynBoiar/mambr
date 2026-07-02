"""Frozen segmentation backbones: SAM 3 (primary), SAM 2 (fallback), mock (tests).

Use :func:`make_backbone` to build one from config; all satisfy the
:class:`zoomtrack.backbone.base.Backbone` interface.
"""

from __future__ import annotations

from zoomtrack.backbone.base import Backbone
from zoomtrack.config import Config

__all__ = ["Backbone", "make_backbone"]


def make_backbone(config: Config | None = None) -> Backbone:
    """Construct the backbone named by ``config.backbone.name``.

    Args:
        config: Project config (``backbone.name`` in ``{"sam3","sam2","mock"}``).

    Returns:
        A frozen :class:`Backbone`.
    """
    cfg = config or Config()
    name = cfg.backbone.name
    if name == "mock":
        from zoomtrack.backbone.mock import MockBackbone

        return MockBackbone(cfg)
    if name == "sam2":
        from zoomtrack.backbone.sam2 import Sam2Backbone

        return Sam2Backbone(cfg)
    if name == "sam3":
        from zoomtrack.backbone.sam3 import Sam3Backbone

        return Sam3Backbone(cfg)
    raise ValueError(f"unknown backbone: {name!r}")
