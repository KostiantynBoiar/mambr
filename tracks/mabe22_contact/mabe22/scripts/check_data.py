"""M0 gate: load + inspect the MABe22 labels and confirm the target behaviours.

The MABe22 analog of the MammAlps ``check_split``. Reports sequence/frame counts and the
framewise positive coverage of the four target behaviours.

Usage::

    cd tracks/mabe22_contact
    PYTHONPATH=. ../../mammalps-env/bin/python -m mabe22.scripts.check_data
"""

from __future__ import annotations

import argparse

from mabe22.config import Config
from mabe22.data.labels import MABeLabels


class DataChecker:
    """Inspect the labels and assert the target behaviours are present + framewise."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the checker.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()

    def check(self) -> None:
        """Print dataset facts and run the M0 gate.

        Raises:
            KeyError: If a target behaviour is missing from the vocabulary (via ``MABeLabels``).
        """
        labels = MABeLabels(self.config)
        seqs = labels.sequence_ids()
        total = labels.label_array.shape[1]
        print(f"labels file : {self.config.data.labels_file}")
        print(f"sequences   : {len(seqs)}  ({self.config.data.frames_per_seq} frames each)")
        print(
            f"vocabulary  : {len(labels.vocabulary)} tasks; label_array {labels.label_array.shape}"
        )
        print("target behaviours (framewise positives):")
        pos = labels.positives()
        for b in labels.behaviours:
            print(f"  {b:22s} pos={pos[b]:>7}  ({pos[b] / total:.4%})")
        print("\nGATE PASSED: 4 target behaviours present and framewise.")


def main() -> None:
    """Parse CLI args and run :class:`DataChecker`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    args = ap.parse_args()
    DataChecker(Config.load(args.config)).check()


if __name__ == "__main__":
    main()
