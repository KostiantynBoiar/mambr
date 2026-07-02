"""Full ZoomTrack: coarse -> select -> zoom -> fuse -> track (Part A cache + Part B) (M4).

Usage::

    PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.run_pipeline --config configs/sam3.yaml
"""

from __future__ import annotations

import argparse

from zoomtrack.config import Config


def main() -> None:
    """Parse args and run the full pipeline over a split."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    _cfg = Config.load(args.config)
    raise NotImplementedError("M4: Pipeline.run over the split -> writer")


if __name__ == "__main__":
    main()
