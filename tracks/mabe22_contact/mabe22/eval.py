"""The locked, model-agnostic MABe22 contact-behaviour scorer (M1).

Given per-frame prediction scores and 0/1 labels for the four target behaviours, report
per-behaviour **F1** (MABe convention) and **average precision** (MammAlps comparability),
plus the macro average. This is the single scorer the whole study reports against; the model
that produced the scores is separate (see ``EVAL_CONTRACT.md``).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from mabe22.config import Config


class MABeScorer:
    """Per-behaviour F1 + AP over the target behaviours."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the scorer.

        Args:
            config: Project config (defaults used when ``None``); supplies the behaviours.
        """
        self.config = config or Config()
        self.behaviours = list(self.config.tasks.behaviours)

    def score(
        self, preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
    ) -> dict[str, dict]:
        """Compute per-behaviour and macro F1 + AP.

        Args:
            preds: ``[N, B]`` float scores (one column per behaviour, same order as
                ``config.tasks.behaviours``).
            labels: ``[N, B]`` 0/1 ground truth.
            threshold: Decision threshold for F1.

        Returns:
            ``{behaviour: {"f1","ap","n_pos","n"}, "macro": {"f1","ap"}}``. AP is ``nan`` for a
            behaviour with no positives in this set.
        """
        out: dict[str, dict] = {}
        for i, b in enumerate(self.behaviours):
            y = labels[:, i].astype(int)
            p = preds[:, i]
            ap = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
            f1 = float(f1_score(y, (p >= threshold).astype(int), zero_division=0))
            out[b] = {"f1": f1, "ap": ap, "n_pos": int(y.sum()), "n": int(len(y))}
        out["macro"] = {
            "f1": float(np.nanmean([out[b]["f1"] for b in self.behaviours])),
            "ap": float(np.nanmean([out[b]["ap"] for b in self.behaviours])),
        }
        return out

    def report(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, dict]:
        """Score and print a per-behaviour table; return the scores."""
        s = self.score(preds, labels)
        for b in self.behaviours:
            r = s[b]
            print(f"  {b:22s} F1={r['f1']:.3f}  AP={r['ap']:.3f}  (pos {r['n_pos']:>7}/{r['n']})")
        print(f"  {'MACRO':22s} F1={s['macro']['f1']:.3f}  AP={s['macro']['ap']:.3f}")
        return s
