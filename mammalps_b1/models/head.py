"""Light, capacity-matched multimodal head + multi-task loss.

Design:

  per-modality projection (Linear -> GELU -> LayerNorm to ``Config.model.fusion_dim``)
  -> SUM the present modalities (keeps trunk input width fixed regardless of how many
     modalities are present, so capacity is identical across video / video+audio /
     +mask -> any gain reflects information, not parameters)
  -> shared MLP trunk
  -> one linear head per task, sized from the :class:`LabelSpace`.

Summation is also modality-dropout / mask ready: adding a ``mask`` modality later only
means registering another projector — no change to the trunk or heads.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from mammalps_b1.config import Config


class ModalityProjector(nn.Module):
    """Project one modality's embedding to the shared fusion width."""

    def __init__(self, in_dim: int, fusion_dim: int) -> None:
        """Initialize the projector.

        Args:
            in_dim: Input embedding dimension for this modality.
            fusion_dim: Shared fusion width to project to.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, fusion_dim), nn.GELU(), nn.LayerNorm(fusion_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` of shape ``[B, in_dim]`` to ``[B, fusion_dim]``."""
        return self.net(x)


class MultiModalHead(nn.Module):
    """Frozen-feature fusion head with one logits output per task."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        task_sizes: Mapping[str, int],
        config: Config | None = None,
    ) -> None:
        """Initialize the head.

        Args:
            modality_dims: ``{modality_name: input_embedding_dim}`` for every modality the
                head *could* receive. Only the modalities present in ``feats`` are fused at
                forward time, so the same module serves video-only and video+audio.
            task_sizes: ``{task: num_classes}`` (e.g. from ``LabelSpace.sizes``).
            config: Project config (defaults used when ``None``); supplies the model dims.
        """
        super().__init__()
        cfg = (config or Config()).model
        self.projectors = nn.ModuleDict(
            {m: ModalityProjector(dim, cfg.fusion_dim) for m, dim in modality_dims.items()}
        )
        self.trunk = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(cfg.hidden_dim, size) for task, size in task_sizes.items()}
        )

    def forward(self, feats: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fuse the present modalities and produce per-task logits.

        Args:
            feats: ``{modality: [B, in_dim]}`` for the modalities present this batch.

        Returns:
            ``{task: [B, num_classes]}`` logits for every task head.

        Raises:
            ValueError: If ``feats`` is empty.
            KeyError: If ``feats`` contains a modality with no registered projector.
        """
        if not feats:
            raise ValueError("MultiModalHead.forward received no modalities")
        unknown = set(feats) - set(self.projectors)
        if unknown:
            raise KeyError(f"No projector for modalities {unknown}; known: {set(self.projectors)}")
        projected = [self.projectors[m](x) for m, x in feats.items()]
        z = torch.stack(projected, dim=0).sum(dim=0)  # sum present modalities
        h = self.trunk(z)
        return {task: head(h) for task, head in self.heads.items()}


class MultiTaskLoss(nn.Module):
    """Weighted CE (multiclass tasks) + BCE (multi-label tasks) from config."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the loss.

        Args:
            config: Project config (defaults used when ``None``); supplies the per-task
                weights and which tasks are multi-label.
        """
        super().__init__()
        cfg = config or Config()
        self.weights = cfg.train.loss_weights
        self.multilabel = set(cfg.tasks.multilabel)

    def forward(
        self, logits: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the weighted multi-task loss.

        Args:
            logits: Per-task logits from :meth:`MultiModalHead.forward`.
            targets: Per-task targets — multi-label tasks expect Float multi-hot
                ``[B, C]``, multiclass tasks expect Long class indices ``[B]``.

        Returns:
            The scalar weighted multi-task loss.
        """
        total = logits[next(iter(logits))].new_zeros(())
        for task, task_logits in logits.items():
            if task in self.multilabel:
                loss = F.binary_cross_entropy_with_logits(task_logits, targets[task])
            else:
                loss = F.cross_entropy(task_logits, targets[task])
            total = total + self.weights.get(task, 1.0) * loss
        return total
