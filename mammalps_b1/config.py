"""Central configuration — the single source of truth for every constant.

Nothing is hardcoded elsewhere: paths, task definitions, model dims, training
hyper-parameters and dataset facts all live here with sensible defaults and are read
through a :class:`Config` instance. Override any field via a YAML file (see
``mammalps_b1/configs/config.example.yaml``) or the ``MAMMALPS_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo root (this file is mammalps_b1/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(env_var: str, default: Path) -> Path:
    """Return ``$env_var`` as a path if set, else ``default``."""
    return Path(os.path.expanduser(os.environ.get(env_var, str(default))))


@dataclass
class PathsConfig:
    """Filesystem locations.

    Attributes:
        data_root: Extracted ``mammalps_v1`` directory (``$MAMMALPS_DATA_ROOT``).
        cache_root: Where cached features live / will be written (``$MAMMALPS_CACHE_ROOT``).
        labels_json: The vendored ``labels_mapping_b1.json`` (the class-space source).
    """

    data_root: Path = field(
        default_factory=lambda: _env_path("MAMMALPS_DATA_ROOT", _REPO_ROOT / "data" / "mammalps_v1")
    )
    cache_root: Path = field(
        default_factory=lambda: _env_path("MAMMALPS_CACHE_ROOT", _REPO_ROOT / "cache")
    )
    labels_json: Path = field(
        default_factory=lambda: _REPO_ROOT / "evaluation" / "labels_mapping_b1.json"
    )


@dataclass
class TasksConfig:
    """Task definitions shared by the labels, head and loss.

    Attributes:
        order: Task order used in the eval columns (ActY, ActN, Spe).
        json_keys: Map each task to its section key in ``labels_mapping_b1.json``.
        multilabel: Tasks scored as multi-label (sigmoid/BCE); the rest are multiclass.
    """

    order: tuple[str, ...] = ("ActY", "ActN", "Spe")
    json_keys: dict[str, str] = field(
        default_factory=lambda: {"ActY": "activities", "ActN": "actions", "Spe": "species"}
    )
    multilabel: tuple[str, ...] = ("ActN",)


@dataclass
class DataConfig:
    """Dataset facts and download locations.

    Attributes:
        splits: Split names.
        expected_rows: Verified clip count per split (test == eval canary).
        total_clips: Verified total clip count.
        action_sep: Separator inside the ``actions`` CSV cell.
        action_none: Padding token to drop from the ``actions`` cell.
        metadata_subdir: Split-CSV directory relative to ``data_root``.
        zip_url: Remote ``mammalps_v1.zip`` (must support HTTP range requests).
        b1_csvs: Split CSV paths inside the zip, fetched without the full download.
    """

    splits: tuple[str, ...] = ("train", "val", "test")
    expected_rows: dict[str, int] = field(
        default_factory=lambda: {"train": 4205, "val": 686, "test": 1244}
    )
    total_clips: int = 6135
    action_sep: str = ";"
    action_none: str = "none"
    metadata_subdir: str = "benchmark_1/metadata"
    zip_url: str = "https://zenodo.org/records/15040901/files/mammalps_v1.zip"
    b1_csvs: tuple[str, ...] = (
        "benchmark_1/metadata/train.csv",
        "benchmark_1/metadata/val.csv",
        "benchmark_1/metadata/test.csv",
    )


@dataclass
class ModelConfig:
    """Fusion-head architecture.

    Attributes:
        fusion_dim: Shared per-modality projection width.
        hidden_dim: Trunk hidden width.
        dropout: Trunk dropout probability.
    """

    fusion_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training hyper-parameters.

    Attributes:
        loss_weights: Per-task multi-task loss weights (Spe / ActY / ActN).
        seed: Random seed for the run.
        num_test_segments: Segment rows emitted per test clip (eval averages them).
    """

    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"Spe": 1.0, "ActY": 2.5, "ActN": 2.0}
    )
    seed: int = 0
    num_test_segments: int = 10


@dataclass
class Config:
    """Top-level configuration aggregating every constant.

    Attributes:
        paths: Filesystem locations.
        tasks: Task definitions.
        data: Dataset facts and download locations.
        model: Fusion-head architecture.
        train: Training hyper-parameters.
        modalities: Active modalities, e.g. ``["video"]`` or ``["video", "audio"]``.
        raw: The raw parsed YAML, for any extra keys.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    modalities: list[str] = field(default_factory=lambda: ["video"])
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
            raw: Parsed YAML mapping; recognised sections are ``paths``, ``tasks``,
                ``data``, ``model``, ``train`` plus top-level ``modalities`` / ``seed``.
        """
        for section in ("paths", "tasks", "data", "model", "train"):
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
        if "modalities" in raw:
            self.modalities = list(raw["modalities"])
        if "seed" in raw:  # convenience top-level alias for train.seed
            self.train.seed = int(raw["seed"])
