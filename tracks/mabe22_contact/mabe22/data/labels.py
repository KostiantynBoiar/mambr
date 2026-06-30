"""Load MABe22 mouse-triplet framewise behaviour labels (M0/M1).

``mouse_submission_labels.npy`` is a pickled dict (the flat-array MABe format):

  - ``vocabulary``: ``list[str]`` of the 13 task names,
  - ``label_array``: ``float32 [n_tasks, total_frames]`` (0/1 for the Discrete behaviour tasks),
  - ``frame_number_map``: ``{sequence_id: (start, end)}`` slicing ``label_array`` columns,
  - ``task_type``: per-task ``"Discrete"`` / ``"Continious"``.

We keep the four target behaviours from :class:`Config.tasks` (``chases``, ``huddles``,
``oral_oral_contact``, ``oral_genital_contact``). Labels are dense (every frame is 0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mabe22.config import Config


@dataclass
class LabeledFrames:
    """Framewise behaviour labels stacked over a set of sequences (the eval unit).

    Attributes:
        labels: ``int8 [N, B]`` 0/1 labels for the ``B`` target behaviours.
        sequence_index: ``int [N]`` — index into the provided sequence list (for grouping).
        frame_index: ``int [N]`` — within-sequence frame number.
    """

    labels: np.ndarray
    sequence_index: np.ndarray
    frame_index: np.ndarray


class MABeLabels:
    """The MABe22 submission-set labels + the four target behaviours."""

    def __init__(self, config: Config | None = None) -> None:
        """Load the labels file described by ``config``.

        Args:
            config: Project config (defaults used when ``None``).

        Raises:
            KeyError: If a configured behaviour is not in the data vocabulary.
        """
        self.config = config or Config()
        path = self.config.paths.data_root / self.config.data.labels_file
        d = np.load(path, allow_pickle=True).item()
        self.vocabulary: list[str] = list(d["vocabulary"])
        self.label_array: np.ndarray = np.asarray(d["label_array"])
        self.frame_number_map: dict[str, tuple[int, int]] = dict(d["frame_number_map"])
        self.task_type: list[str] = list(d["task_type"])
        self.behaviours: list[str] = list(self.config.tasks.behaviours)
        missing = [b for b in self.behaviours if b not in self.vocabulary]
        if missing:
            raise KeyError(f"behaviours not in vocabulary {self.vocabulary}: {missing}")
        self.behaviour_idx: list[int] = [self.vocabulary.index(b) for b in self.behaviours]

    def sequence_ids(self) -> list[str]:
        """All labelled sequence ids."""
        return list(self.frame_number_map)

    def positives(self) -> dict[str, int]:
        """Number of positive frames per target behaviour (over the whole set)."""
        return {
            b: int((self.label_array[i] == 1).sum())
            for b, i in zip(self.behaviours, self.behaviour_idx, strict=True)
        }

    def frames_for(self, seq_ids: list[str]) -> LabeledFrames:
        """Stack framewise target-behaviour labels for ``seq_ids``.

        Args:
            seq_ids: The sequences to include (e.g. one split).

        Returns:
            A :class:`LabeledFrames` with ``labels`` ``[N, B]`` plus per-frame sequence/frame
            indices.
        """
        rows = self.behaviour_idx
        labs, sidx, fidx = [], [], []
        for s, sid in enumerate(seq_ids):
            start, end = self.frame_number_map[sid]
            n = end - start
            labs.append(self.label_array[rows, start:end].T)  # [n, B]
            sidx.append(np.full(n, s, dtype=np.int32))
            fidx.append(np.arange(n, dtype=np.int32))
        return LabeledFrames(
            labels=np.concatenate(labs).astype(np.int8),
            sequence_index=np.concatenate(sidx),
            frame_index=np.concatenate(fidx),
        )
