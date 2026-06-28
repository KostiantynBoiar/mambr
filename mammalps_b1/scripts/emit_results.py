"""Emit a results.txt for eval_b1.py — the format canary.

Writes one tab-separated row per (test clip x segment) with **real labels** and **random
predictions**, in the exact layout the scorer expects::

    id  sample_id  test_segment_id  ActY_preds  ActN_preds  Spe_preds  ActY_label  ActN_label  Spe_label

preds then labels, each in ``Config.tasks.order`` (ActY, ActN, Spe); every vector is a
bracketed comma-joined float list ``[a,b,c,...]``. With ``num_test_segments`` rows per clip,
the scorer groups by ``sample_id`` down to 1244 rows and runs its assertion.

This is also the skeleton for the inference script — only ``_random_preds`` is replaced by
the trained head's outputs.

Usage::

    mammalps-env/bin/python -m mammalps_b1.scripts.emit_results [--config c.yaml] [--split test] [--out path]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mammalps_b1.config import Config
from mammalps_b1.data.labels import LabelSpace
from mammalps_b1.data.splits import SplitLoader


class ResultsEmitter:
    """Write a contract-correct results.txt with real labels and random predictions."""

    def __init__(
        self,
        config: Config | None = None,
        labels: LabelSpace | None = None,
        loader: SplitLoader | None = None,
    ) -> None:
        """Initialize the emitter.

        Args:
            config: Project config (defaults used when ``None``).
            labels: A pre-built label space; constructed from ``config`` when ``None``.
            loader: A pre-built split loader; constructed from ``config`` when ``None``.
        """
        self.config = config or Config()
        self.labels = labels or LabelSpace(self.config)
        self.loader = loader or SplitLoader(self.config, self.labels)

    @staticmethod
    def _fmt_vec(v: np.ndarray) -> str:
        """Serialise a vector as ``[a,b,c]`` (no spaces/tabs), parsable by the scorer."""
        return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v, dtype=float)) + "]"

    def _random_preds(self, task: str, rng: np.random.Generator) -> np.ndarray:
        """Random score vector for ``task``: sigmoid for multi-label, softmax otherwise."""
        logits = rng.standard_normal(self.labels.sizes[task])
        if task in self.config.tasks.multilabel:
            return 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> per-class [0, 1]
        exp = np.exp(logits - logits.max())  # softmax -> sums to 1
        return exp / exp.sum()

    def emit(self, split: str = "test", out_path: Path | str | None = None) -> Path:
        """Write the results file for ``split`` and return its path.

        Args:
            split: Which split to emit (default ``"test"`` — the one the scorer asserts on).
            out_path: Output path; defaults to ``results_canary_<split>.txt``.

        Returns:
            The path written.
        """
        out = Path(out_path) if out_path else Path(f"results_canary_{split}.txt")
        records = self.loader.load(split)
        rng = np.random.default_rng(self.config.train.seed)
        tasks = self.config.tasks.order
        n_seg = self.config.train.num_test_segments

        row_id = 0
        with open(out, "w") as f:
            for rec in records:
                for seg in range(n_seg):
                    preds = [self._fmt_vec(self._random_preds(t, rng)) for t in tasks]
                    lbls = [self._fmt_vec(rec.label(t)) for t in tasks]
                    f.write(
                        "\t".join([str(row_id), str(rec.sample_id), str(seg), *preds, *lbls]) + "\n"
                    )
                    row_id += 1

        print(f"wrote {out}  ({row_id} rows = {len(records)} clips x {n_seg} segments)")
        return out


def main() -> None:
    """Parse CLI args and run :class:`ResultsEmitter`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="output path (default results_canary_<split>.txt)")
    args = ap.parse_args()
    ResultsEmitter(Config.load(args.config)).emit(args.split, args.out)


if __name__ == "__main__":
    main()
