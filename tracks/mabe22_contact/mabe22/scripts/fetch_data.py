"""Download MABe22 mouse-triplet pose+label files from Caltech (M0).

For M0/M1 we need only the labels file (``mouse_submission_labels.npy``, ~171 MB). Keypoints
(~949 MB, M4) and the video zip (~3.5 GB, M2) are fetched later. Caltech serves each file via a
302 redirect to a short-lived presigned URL; ``urllib`` follows it automatically.

Usage::

    cd tracks/mabe22_contact
    PYTHONPATH=. ../../mammalps-env/bin/python -m mabe22.scripts.fetch_data [--files labels keypoints]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from mabe22.config import Config

_NAME = "MABe22 mouse-triplet fetch"


class MABeFetcher:
    """Download chosen MABe22 files into ``config.paths.data_root``."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the fetcher.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()

    def _filename(self, key: str) -> str:
        """Map a logical key (labels/keypoints/video) to the Caltech filename."""
        d = self.config.data
        return {
            "labels": d.labels_file,
            "keypoints": d.keypoints_file,
            "video": d.video_zip,
        }[key]

    def fetch(self, keys: list[str]) -> list[Path]:
        """Download the files for ``keys`` (skipping ones already present).

        Args:
            keys: Logical names, e.g. ``["train"]`` (the M0/M1 default).

        Returns:
            The local paths of the requested files.
        """
        data_root = self.config.paths.data_root
        data_root.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for key in keys:
            name = self._filename(key)
            dest = data_root / name
            if dest.exists():
                print(f"  have {name} ({dest.stat().st_size:,} B)")
            else:
                url = f"{self.config.data.base_url}/{name}"
                print(f"  downloading {name} ...")
                tmp = dest.with_suffix(dest.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)  # follows the 302 redirect
                tmp.rename(dest)
                print(f"  wrote {dest} ({dest.stat().st_size:,} B)")
            out.append(dest)
        return out


def main() -> None:
    """Parse CLI args and run :class:`MABeFetcher`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    ap.add_argument(
        "--files",
        nargs="+",
        default=["labels"],
        choices=["labels", "keypoints", "video"],
        help="which files to fetch (default: labels = enough for M0/M1)",
    )
    args = ap.parse_args()
    MABeFetcher(Config.load(args.config)).fetch(args.files)


if __name__ == "__main__":
    main()
