"""Scaffold sanity — the package imports cleanly and every config parses.

Passes at scaffold time (stubs raise ``NotImplementedError`` only when *called*, not on import).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zoomtrack.config import Config


def test_import_package() -> None:
    """The top-level package imports and exposes a version."""
    import zoomtrack

    assert zoomtrack.__version__


def test_import_all_modules() -> None:
    """Every module imports without error (no import-time side effects in the stubs)."""
    import zoomtrack.backbone  # noqa: F401
    import zoomtrack.backbone.base  # noqa: F401
    import zoomtrack.backbone.mock  # noqa: F401
    import zoomtrack.backbone.sam2  # noqa: F401
    import zoomtrack.backbone.sam3  # noqa: F401
    import zoomtrack.data.dataset  # noqa: F401
    import zoomtrack.eval  # noqa: F401
    import zoomtrack.fusion  # noqa: F401
    import zoomtrack.geometry  # noqa: F401
    import zoomtrack.masks  # noqa: F401
    import zoomtrack.pipeline  # noqa: F401
    import zoomtrack.selector.features  # noqa: F401
    import zoomtrack.selector.head  # noqa: F401
    import zoomtrack.tile_selector  # noqa: F401
    import zoomtrack.tracking  # noqa: F401
    import zoomtrack.train_selector  # noqa: F401
    import zoomtrack.utils.device  # noqa: F401
    import zoomtrack.utils.seed  # noqa: F401
    import zoomtrack.writer  # noqa: F401


def test_default_config() -> None:
    """Dataclass defaults are sane."""
    cfg = Config()
    assert cfg.tiling.tile == 512
    assert cfg.backbone.name == "mock"
    assert cfg.selector.tile_budget >= 1


@pytest.mark.parametrize("name", ["default", "mock", "sam2", "sam3"])
def test_configs_parse(name: str) -> None:
    """Each shipped YAML overlays onto the config without error."""
    path = Path(__file__).resolve().parents[1] / "zoomtrack" / "configs" / f"{name}.yaml"
    cfg = Config.load(path)
    assert cfg.selector.tile_budget >= 1
    assert cfg.fusion.mode in {"union", "prefer_tile"}
