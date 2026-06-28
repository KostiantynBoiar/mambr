"""Verify the Benchmark-I splits and print label distributions.

GATE: the test split must equal ``Config.data.expected_rows["test"]`` (1244 — the number
``eval_b1.py`` asserts on); the total must equal ``Config.data.total_clips`` (6135).

Usage::

    mammalps-env/bin/python -m mammalps_b1.scripts.check_split [--config path/to.yaml]
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from mammalps_b1.config import Config
from mammalps_b1.data.labels import LabelSpace
from mammalps_b1.data.splits import ClipRecord, SplitLoader


class SplitChecker:
    """Load all splits, assert the expected counts, and report test distributions."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the checker.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()
        self.labels = LabelSpace(self.config)
        self.loader = SplitLoader(self.config, self.labels)

    def check(self) -> None:
        """Run the gate and print distributions.

        Raises:
            AssertionError: If any split count (or the total) differs from the expectation.
        """
        data = self.config.data
        counts: dict[str, int] = {}
        records: dict[str, list[ClipRecord]] = {}
        for split in data.splits:
            recs = self.loader.load(split)
            records[split], counts[split] = recs, len(recs)
            print(f"{split:>5}: {len(recs):>5} clips  (expected {data.expected_rows[split]})")

        total = sum(counts.values())
        print(f"total: {total:>5} clips  (expected {data.total_clips})")

        for split in data.splits:
            assert counts[split] == data.expected_rows[split], (
                f"{split}={counts[split]} != {data.expected_rows[split]}"
            )
        assert total == data.total_clips, f"total is {total}, expected {data.total_clips}"

        self._print_distributions(records["test"])
        print("\nGATE PASSED: test==1244, total==6135, all labels in-vocab.")

    def _print_distributions(self, test: list[ClipRecord]) -> None:
        """Print per-task label distributions for the test split.

        Args:
            test: The test-split clip records.
        """
        names = self.labels.class_lists()
        spe = Counter(names["Spe"][int(np.argmax(r.spe_onehot))] for r in test)
        acty = Counter(names["ActY"][int(np.argmax(r.acty_onehot))] for r in test)
        actn = Counter(names["ActN"][i] for r in test for i in np.nonzero(r.actn_multihot)[0])
        print("\ntest species   :", dict(spe))
        print("test activities:", dict(sorted(acty.items(), key=lambda kv: -kv[1])))
        print("test actions   :", dict(sorted(actn.items(), key=lambda kv: -kv[1])))


def main() -> None:
    """Parse CLI args and run :class:`SplitChecker`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    args = ap.parse_args()
    SplitChecker(Config.load(args.config)).check()


if __name__ == "__main__":
    main()
