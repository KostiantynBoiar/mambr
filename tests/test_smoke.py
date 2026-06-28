"""Data-free smoke test (toolchain check, no data or GPU needed).

Confirms the toolchain end-to-end before any real data exists:

  - the config + label space load and the head builds from them,
  - a forward pass yields the right per-task shapes (sizes from the label space),
  - the multi-task loss (CE + CE + BCE) is finite and back-propagates,
  - capacity is identical across configs (same trunk/head params, since summation fusion
    keeps the trunk fixed — only the per-modality projectors differ).

Run: ``mammalps-env/bin/pytest -q``
"""

from __future__ import annotations

import torch

from mammalps_b1.config import Config
from mammalps_b1.data.labels import LabelSpace
from mammalps_b1.models.head import MultiModalHead, MultiTaskLoss
from mammalps_b1.utils.seed import set_seed

VIDEO_DIM, AUDIO_DIM, BATCH = 768, 768, 8

_CONFIG = Config()
_SIZES = LabelSpace(_CONFIG).sizes  # {"ActY": 11, "ActN": 19, "Spe": 5}


def _random_targets(batch: int) -> dict[str, torch.Tensor]:
    """Build random per-task targets (Long indices for multiclass, multi-hot for ActN)."""
    multilabel = set(_CONFIG.tasks.multilabel)
    targets: dict[str, torch.Tensor] = {}
    for task, size in _SIZES.items():
        if task in multilabel:
            targets[task] = (torch.rand(batch, size) > 0.5).float()
        else:
            targets[task] = torch.randint(0, size, (batch,))
    return targets


def _build_head(modality_dims: dict[str, int]) -> MultiModalHead:
    """Build a head for the given modalities using the default config + label sizes."""
    return MultiModalHead(modality_dims, _SIZES, _CONFIG)


def test_forward_shapes_and_backward() -> None:
    """Forward yields the right shapes and the loss back-propagates into the head."""
    set_seed(0)
    head = _build_head({"video": VIDEO_DIM, "audio": AUDIO_DIM})
    feats = {"video": torch.randn(BATCH, VIDEO_DIM), "audio": torch.randn(BATCH, AUDIO_DIM)}

    logits = head(feats)
    for task, size in _SIZES.items():
        assert logits[task].shape == (BATCH, size)

    loss = MultiTaskLoss(_CONFIG)(logits, _random_targets(BATCH))
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow through the trainable head
    assert head.heads["Spe"].weight.grad is not None


def test_video_only_subset_runs() -> None:
    """The same head serves video-only by passing a single-modality dict."""
    set_seed(0)
    head = _build_head({"video": VIDEO_DIM, "audio": AUDIO_DIM})
    logits = head({"video": torch.randn(BATCH, VIDEO_DIM)})
    assert logits["ActN"].shape == (BATCH, _SIZES["ActN"])


def test_capacity_identical_across_configs() -> None:
    """Trunk + task-head params are identical regardless of modality set (summation)."""
    head_v = _build_head({"video": VIDEO_DIM})
    head_va = _build_head({"video": VIDEO_DIM, "audio": AUDIO_DIM})

    def core_params(m: MultiModalHead) -> int:
        return sum(p.numel() for n, p in m.named_parameters() if not n.startswith("projectors"))

    assert core_params(head_v) == core_params(head_va)


def test_mps_or_cpu_device() -> None:
    """A tensor + head run on MPS when available, else CPU (env gate)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    head = _build_head({"video": VIDEO_DIM}).to(device)
    out = head({"video": torch.randn(4, VIDEO_DIM, device=device)})
    assert out["Spe"].device.type == device
