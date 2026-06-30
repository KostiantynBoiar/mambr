"""M1 format canary: score dummy predictions end-to-end (mirror MammAlps emit_results).

Builds the leakage-safe test split, takes the **real** labels with **random** prediction
scores, and runs :class:`MABeScorer`. Proves the eval plumbing (split + metric) is correct
before any geometry/model exists. Values come out near-random by design.

Usage::

    cd tracks/mabe22_contact
    PYTHONPATH=. ../../mammalps-env/bin/python -m mabe22.scripts.emit_dummy
"""

from __future__ import annotations

import argparse

import numpy as np

from mabe22.config import Config
from mabe22.data.labels import MABeLabels
from mabe22.data.split import make_sequence_split
from mabe22.eval import MABeScorer


class DummyEmitter:
    """Run the scorer on random predictions over the real test split."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the emitter.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()

    def run(self) -> None:
        """Build the split, score random preds, and confirm the canary.

        Raises:
            AssertionError: If the split is not leakage-safe (shared sequences).
        """
        labels = MABeLabels(self.config)
        split = make_sequence_split(labels.sequence_ids(), self.config)
        assert not (set(split["train"]) & set(split["test"])), "split leaks sequences"

        test = labels.frames_for(split["test"])
        rng = np.random.default_rng(self.config.seed)
        preds = rng.random(test.labels.shape).astype(np.float32)

        print(
            f"split (by sequence): train {len(split['train'])} / "
            f"val {len(split['val'])} / test {len(split['test'])}"
        )
        print(f"test frames: {len(test.labels):,}")
        print("dummy (random) predictions scored:")
        MABeScorer(self.config).report(preds, test.labels)
        print("\nCANARY OK: scorer ran end-to-end on dummy preds.")


def main() -> None:
    """Parse CLI args and run :class:`DummyEmitter`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    args = ap.parse_args()
    DummyEmitter(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
