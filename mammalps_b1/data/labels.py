"""Class spaces for Benchmark I, encapsulated as :class:`LabelSpace`.

The class *order* follows ``Config.tasks.order`` (ActY, ActN, Spe) — the order the eval
argmaxes/vstacks in. Sizes are derived from the JSON, so the mapping file is the single
source of truth for the class spaces.
"""

from __future__ import annotations

import json

import numpy as np

from mammalps_b1.config import Config


class LabelSpace:
    """The Benchmark-I class spaces and one-hot/multi-hot encoders."""

    def __init__(self, config: Config | None = None) -> None:
        """Load the label mapping described by ``config``.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()
        with open(self.config.paths.labels_json) as f:
            self._mapping: dict[str, dict[str, int]] = json.load(f)
        self.tasks: tuple[str, ...] = self.config.tasks.order
        self._json_keys: dict[str, str] = self.config.tasks.json_keys

    @property
    def sizes(self) -> dict[str, int]:
        """Number of classes per task, derived from the mapping."""
        return {task: len(self._mapping[self._json_keys[task]]) for task in self.tasks}

    def name_to_index(self) -> dict[str, dict[str, int]]:
        """Build a per-task ``{class_name: class_id}`` lookup.

        Returns:
            ``{task: {class_name: class_id}}`` for each task in :attr:`tasks`.
        """
        return {task: dict(self._mapping[self._json_keys[task]]) for task in self.tasks}

    def class_lists(self) -> dict[str, list[str]]:
        """Build per-task class-name lists ordered by class id (the vector layout).

        Returns:
            ``{task: [class_name ordered by class_id]}`` for each task in :attr:`tasks`.
        """
        out: dict[str, list[str]] = {}
        for task in self.tasks:
            d = self._mapping[self._json_keys[task]]
            out[task] = [name for name, _ in sorted(d.items(), key=lambda kv: kv[1])]
        return out

    def one_hot(self, class_name: str, task: str) -> np.ndarray:
        """Encode a single class as a one-hot vector.

        Args:
            class_name: The class name to set.
            task: One of :attr:`tasks`.

        Returns:
            A ``float32`` one-hot vector of length ``sizes[task]``.
        """
        v = np.zeros(self.sizes[task], dtype=np.float32)
        v[self._mapping[self._json_keys[task]][class_name]] = 1.0
        return v

    def multi_hot(self, class_names: list[str], task: str) -> np.ndarray:
        """Encode several classes as a multi-hot vector.

        Args:
            class_names: The class names to set (may be empty).
            task: One of :attr:`tasks`.

        Returns:
            A ``float32`` multi-hot vector of length ``sizes[task]``.
        """
        index = self._mapping[self._json_keys[task]]
        v = np.zeros(self.sizes[task], dtype=np.float32)
        for name in class_names:
            v[index[name]] = 1.0
        return v
