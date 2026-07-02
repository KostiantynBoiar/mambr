"""Bucket predictions small/large and invoke veval; report pHOTA/pDetA/pAssA (M0/M5).

Usage::

    PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.score --pred pred.json --gt gt.json
"""

from __future__ import annotations

import argparse

from zoomtrack.config import Config
from zoomtrack.eval import Evaluator


def main() -> None:
    """Parse args and score predictions with veval, per subset."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--pred", required=True, help="predicted VEval JSON")
    ap.add_argument("--gt", required=True, help="ground-truth VEval JSON")
    args = ap.parse_args()
    Evaluator(Config.load(args.config)).score(args.pred, args.gt)


if __name__ == "__main__":
    main()
