"""Train the learned tile-selector head (thin CLI over :mod:`zoomtrack.train_selector`) (M3).

Usage::

    PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.train_selector --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from zoomtrack.config import Config


def main() -> None:
    """Parse args and train the selector head."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    _cfg = Config.load(args.config)
    raise NotImplementedError("M3: zoomtrack.train_selector.train(...)")


if __name__ == "__main__":
    main()
