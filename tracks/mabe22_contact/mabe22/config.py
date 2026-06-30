"""Central configuration for the MABe22 contact-geometry track.

Single source of truth for every constant (paths, dataset files, the four target
behaviours, the eval policy). Defaults are overridable via a YAML file or ``MABE_*``
environment variables. Mirrors the MammAlps track's ``Config`` pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# track root (this file is tracks/mabe22_contact/mabe22/config.py)
_TRACK_ROOT = Path(__file__).resolve().parents[1]


def _env_path(env_var: str, default: Path) -> Path:
    """Return ``$env_var`` as a path if set, else ``default``."""
    return Path(os.path.expanduser(os.environ.get(env_var, str(default))))


@dataclass
class PathsConfig:
    """Filesystem locations.

    Attributes:
        data_root: Where the MABe22 ``.npy`` files live (``$MABE_DATA_ROOT``).
        cache_root: Where cached features/masks live (``$MABE_CACHE_ROOT``).
    """

    data_root: Path = field(
        default_factory=lambda: _env_path("MABE_DATA_ROOT", _TRACK_ROOT / "data")
    )
    cache_root: Path = field(
        default_factory=lambda: _env_path("MABE_CACHE_ROOT", _TRACK_ROOT / "cache")
    )


@dataclass
class DataConfig:
    """MABe22 mouse-triplet dataset facts and download locations.

    The study uses the **submission set** (the one with video + keypoints + labels);
    confirmed at M0. ``mouse_submission_labels.npy`` is the flat-array MABe format
    (``vocabulary`` / ``label_array[task, frame]`` / ``frame_number_map`` {seq: (start,end)}).
    The ``mouse_triplet_train/test`` files are the representation-learning challenge data
    (no video; only chases+lights labelled) — not used for the supervised contact study.

    Attributes:
        base_url: Caltech record base URL; files are ``<base_url>/<name>``.
        labels_file: Framewise task labels (+ vocabulary, frame_number_map). Used for M0/M1.
        keypoints_file: Keypoints for the same set (needed from M4: hulls + SAM2 prompts).
        video_zip: 224x224 grayscale video (needed only from the masking pilot, M2).
        fps: Frame rate (Hz).
        frames_per_seq: Frames per clip (1 min @ 30 Hz).
        n_mice: Mice per clip.
        n_keypoints: Keypoints per mouse.
    """

    base_url: str = "https://data.caltech.edu/records/rdsa8-rde65/files"
    labels_file: str = "mouse_submission_labels.npy"
    keypoints_file: str = "mouse_submission_keypoints.npy"
    video_zip: str = "mouse_submission_videos_resized_224.zip"
    fps: int = 30
    frames_per_seq: int = 1800
    n_mice: int = 3
    n_keypoints: int = 12


@dataclass
class TasksConfig:
    """The four target framewise social/contact behaviours (exact MABe22 vocabulary names).

    Attributes:
        behaviours: ``label_array`` task names — chase / huddle / oral-oral (≈oral contact) /
            oral-genital. (Vocabulary also has approaches, close, contact, oral_ear_contact,
            watching — available as extra contact tasks if wanted later.)
    """

    behaviours: tuple[str, ...] = (
        "chases",
        "huddles",
        "oral_oral_contact",
        "oral_genital_contact",
    )


@dataclass
class EvalConfig:
    """The locked label/eval contract (see EVAL_CONTRACT.md, set at M1).

    Attributes:
        split_by: Grouping key for a leakage-safe split (never random frame/window).
        val_frac: Fraction of sequences held out for validation.
        test_frac: Fraction of sequences held out for test.
        metrics: Reported per-behaviour metrics (F1 = MABe; AP = MammAlps comparability).
    """

    split_by: str = "sequence"
    val_frac: float = 0.15
    test_frac: float = 0.15
    metrics: tuple[str, ...] = ("f1", "ap")


@dataclass
class Config:
    """Top-level configuration aggregating every constant.

    Attributes:
        paths: Filesystem locations.
        data: Dataset facts and download locations.
        tasks: Target behaviours.
        eval: Label/eval contract policy.
        seed: Random seed.
        raw: Raw parsed YAML, for any extra keys.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        """Build a config from defaults, then overlay a YAML file if given.

        Args:
            path: Optional YAML config path; defaults-only when ``None``.

        Returns:
            The resolved :class:`Config`.
        """
        cfg = cls()
        if path is not None:
            with open(path) as f:
                cfg.raw = yaml.safe_load(f) or {}
            cfg._apply(cfg.raw)
        return cfg

    def _apply(self, raw: dict[str, Any]) -> None:
        """Overlay parsed YAML onto the nested sections in place.

        Args:
            raw: Parsed YAML mapping; recognised sections are ``paths``, ``data``,
                ``tasks``, ``eval`` plus a top-level ``seed``.
        """
        for section in ("paths", "data", "tasks", "eval"):
            values = raw.get(section)
            if not values:
                continue
            sub = getattr(self, section)
            for key, value in values.items():
                if not hasattr(sub, key):
                    continue
                if section == "paths":
                    value = Path(os.path.expanduser(str(value)))
                setattr(sub, key, value)
        if "seed" in raw:
            self.seed = int(raw["seed"])
