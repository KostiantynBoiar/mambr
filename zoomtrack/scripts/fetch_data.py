"""Acquire + verify SA-FARI into ``data/`` (M0).

Two downloads: media (6 fps ``JPEGImages_6fps`` frames from the CXL bucket) and the annotation JSONs
(Hugging Face). Verifies frame/label alignment. See ``SA-FARI_dataset_reference.md``.

Usage::

    PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.fetch_data [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse

from zoomtrack.config import Config


def main() -> None:
    """Parse args and fetch SA-FARI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    _cfg = Config.load(args.config)
    raise NotImplementedError("M0: download media (CXL) + annotations (HF), verify alignment")


if __name__ == "__main__":
    main()
