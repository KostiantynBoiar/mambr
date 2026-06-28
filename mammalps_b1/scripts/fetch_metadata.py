"""Fetch the small Benchmark-I split CSVs WITHOUT downloading the 88 GB zip.

Zenodo serves the archive with HTTP range support, so ``remotezip`` reads only the zip's
central directory + the few KB of each CSV. Output::

    <data_root>/benchmark_1/metadata/{train,val,test}.csv

Usage::

    mammalps-env/bin/python -m mammalps_b1.scripts.fetch_metadata [--config path/to.yaml]
"""

from __future__ import annotations

import argparse

from remotezip import RemoteZip

from mammalps_b1.config import Config


class MetadataFetcher:
    """Extract the Benchmark-I split CSVs from the remote zip via HTTP range requests."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the fetcher.

        Args:
            config: Project config (defaults used when ``None``); supplies the zip URL,
                the in-zip CSV paths and the destination ``data_root``.
        """
        self.config = config or Config()

    def fetch(self) -> None:
        """Download just the split CSVs into ``config.paths.data_root``."""
        data_root = self.config.paths.data_root
        data_root.mkdir(parents=True, exist_ok=True)
        with RemoteZip(self.config.data.zip_url) as z:
            for name in self.config.data.b1_csvs:
                z.extract(name, path=data_root)
                out = data_root / name
                print(f"fetched {out}  ({out.stat().st_size:,} B)")


def main() -> None:
    """Parse CLI args and run :class:`MetadataFetcher`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    args = ap.parse_args()
    MetadataFetcher(Config.load(args.config)).fetch()


if __name__ == "__main__":
    main()
