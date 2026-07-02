"""ZoomTrack — coarse-to-fine, tile-refined promptable segmentation and tracking.

Targets SA-FARI's small-animal accuracy gap: a cheap coarse pass on the downscaled frame, a
learned selection of tiles that likely hide missed animals, full-resolution re-segmentation of
those tiles, cross-scale fusion into consistent per-frame instances, and temporal linking into
masklets. The heavy segmentation backbone is frozen; only a small tile-selector (and optionally a
fusion head) is trained.

See ``.claude/CLAUDE.md`` for the job spec and ``ROADMAP.md`` for the M0-M6 build plan.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
