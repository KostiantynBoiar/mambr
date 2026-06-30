"""Parse the Benchmark-I split CSVs into clip records, via :class:`SplitLoader`.

The CSVs live at ``<data_root>/<metadata_subdir>/{split}.csv`` and are **comma-separated
with a header**::

    video_path,start_s,end_s,activity,actions,species

  - ``video_path``: ``"<track_dir>/<clip_id>.mp4"`` (``clip_id`` = filename stem)
  - ``start_s``: always ``0.0`` (clips are pre-cut)
  - ``end_s``: clip duration in seconds (can be < 1 s — short clips need frame looping
    when features are extracted)
  - ``activity``: one ActY class -> one-hot
  - ``actions``: 1-2 ActN classes, ``Config.data.action_sep``-separated, padded with
    ``Config.data.action_none`` -> multi-hot
  - ``species``: one Spe class -> one-hot

``sample_id`` is the 0-based row index within a split's CSV; the eval groups test rows by
it (README: "sample_id ... corresponding to the row ids in test.csv").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mammalps_b1.config import Config
from mammalps_b1.data.labels import LabelSpace


@dataclass
class ClipRecord:
    """One Benchmark-I clip with its parsed label vectors.

    Attributes:
        clip_id: The mp4 filename stem (e.g. ``S1_C3_E154_V0066_ID1_T1_c0``).
        video_path: Path to the clip mp4 relative to the benchmark root.
        start_s: Clip start time in seconds (always ``0.0``).
        end_s: Clip end time in seconds (clip duration).
        split: One of ``Config.data.splits``.
        sample_id: 0-based row index within the split CSV (the eval groups by this).
        spe_onehot: Species one-hot.
        acty_onehot: Activity one-hot.
        actn_multihot: Action multi-hot.
    """

    clip_id: str
    video_path: str
    start_s: float
    end_s: float
    split: str
    sample_id: int
    spe_onehot: np.ndarray
    acty_onehot: np.ndarray
    actn_multihot: np.ndarray

    def label(self, task: str) -> np.ndarray:
        """Return this clip's label vector for ``task``.

        Args:
            task: One of ``"ActY"``, ``"ActN"`` or ``"Spe"``.

        Returns:
            The matching one-hot/multi-hot vector.

        Raises:
            KeyError: If ``task`` is not a known task.
        """
        vectors = {"ActY": self.acty_onehot, "ActN": self.actn_multihot, "Spe": self.spe_onehot}
        if task not in vectors:
            raise KeyError(f"unknown task {task!r}; expected one of {tuple(vectors)}")
        return vectors[task]


class SplitLoader:
    """Load Benchmark-I split CSVs into clip records with label vectors."""

    def __init__(self, config: Config | None = None, labels: LabelSpace | None = None) -> None:
        """Initialize the loader.

        Args:
            config: Project config (defaults used when ``None``).
            labels: A pre-built label space; constructed from ``config`` when ``None``.
        """
        self.config = config or Config()
        self.labels = labels or LabelSpace(self.config)

    @property
    def metadata_dir(self) -> Path:
        """Directory holding the split CSVs (``<data_root>/<metadata_subdir>``)."""
        return self.config.paths.data_root / self.config.data.metadata_subdir

    def parse_actions(self, actions_field: str) -> list[str]:
        """Split the ``actions`` cell into real action names, dropping the pad token.

        Args:
            actions_field: Raw cell value, e.g. ``"walking;none"`` or ``"running;jumping"``.

        Returns:
            The non-pad action names (length 1 or 2).
        """
        sep, none = self.config.data.action_sep, self.config.data.action_none
        return [a for a in str(actions_field).split(sep) if a and a != none]

    def load(self, split: str) -> list[ClipRecord]:
        """Load one split CSV into a list of :class:`ClipRecord`.

        Args:
            split: One of ``Config.data.splits``.

        Returns:
            One :class:`ClipRecord` per CSV row, in file order.

        Raises:
            ValueError: If ``split`` is not a known split.
        """
        if split not in self.config.data.splits:
            raise ValueError(f"split must be one of {self.config.data.splits}, got {split!r}")
        df = pd.read_csv(self.metadata_dir / f"{split}.csv").reset_index(drop=True)

        records: list[ClipRecord] = []
        for sample_id, row in df.iterrows():
            records.append(
                ClipRecord(
                    clip_id=Path(row["video_path"]).stem,
                    video_path=str(row["video_path"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    split=split,
                    sample_id=int(sample_id),
                    spe_onehot=self.labels.one_hot(row["species"], "Spe"),
                    acty_onehot=self.labels.one_hot(row["activity"], "ActY"),
                    actn_multihot=self.labels.multi_hot(self.parse_actions(row["actions"]), "ActN"),
                )
            )
        return records

    def load_df(self, split: str) -> pd.DataFrame:
        """Load one split as a DataFrame with label vectors stored as lists.

        Lists (rather than numpy arrays) keep the frame parquet-friendly for the S4 manifest.

        Args:
            split: One of ``Config.data.splits``.

        Returns:
            A DataFrame with one row per clip and label vectors as Python lists.
        """
        return pd.DataFrame(
            [
                {
                    "clip_id": r.clip_id,
                    "video_path": r.video_path,
                    "start_s": r.start_s,
                    "end_s": r.end_s,
                    "split": r.split,
                    "sample_id": r.sample_id,
                    "spe_onehot": r.spe_onehot.tolist(),
                    "acty_onehot": r.acty_onehot.tolist(),
                    "actn_multihot": r.actn_multihot.tolist(),
                }
                for r in self.load(split)
            ]
        )
