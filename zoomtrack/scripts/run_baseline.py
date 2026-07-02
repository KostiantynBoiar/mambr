"""Coarse-only baseline: segment the downscaled frame, track, write VEval JSON (M0/M2).

The reference the whole method is measured against. Usage::

    PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.run_baseline --config configs/sam2.yaml
"""

from __future__ import annotations

import argparse

from zoomtrack.config import Config


def main() -> None:
    """Parse args and run the coarse-only baseline over a split."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    _cfg = Config.load(args.config)
    raise NotImplementedError("M0: Pipeline.run_coarse_only over the split -> writer")


if __name__ == "__main__":
    main()
