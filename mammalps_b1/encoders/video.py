"""Frozen VideoMAE video encoder: clip mp4 -> one pooled embedding.

Decodes a fixed number of frames from a clip, runs them through a frozen VideoMAE, and
mean-pools the token sequence into a single vector. Never trained (no gradients).
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from mammalps_b1.config import Config
from mammalps_b1.utils.device import get_device


class VideoEncoder:
    """Frozen VideoMAE wrapper producing one embedding per clip."""

    def __init__(self, config: Config | None = None) -> None:
        """Load the frozen VideoMAE checkpoint and image processor.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()
        cfg = self.config.encoders
        self.num_frames = cfg.num_frames
        self.device = get_device(cfg.device)
        self.processor = VideoMAEImageProcessor.from_pretrained(cfg.video_ckpt)
        self.model = VideoMAEModel.from_pretrained(cfg.video_ckpt).to(self.device).eval()
        self.model.requires_grad_(False)

    @property
    def dim(self) -> int:
        """Embedding dimension (the backbone hidden size)."""
        return int(self.model.config.hidden_size)

    def _read_frames(self, clip_path: Path | str) -> np.ndarray:
        """Decode and uniformly sample ``num_frames`` RGB frames from the clip.

        Short clips (fewer decoded frames than ``num_frames``) are looped/padded by
        index sampling, so a 1-frame clip simply repeats that frame.

        Args:
            clip_path: Path to the clip mp4.

        Returns:
            A ``uint8`` array of shape ``[num_frames, H, W, 3]``.
        """
        with av.open(str(clip_path)) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if not frames:
            raise RuntimeError(f"no decodable frames in {clip_path}")
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        return np.stack([frames[i] for i in idx])

    def embed(self, clip_path: Path | str) -> np.ndarray:
        """Encode a clip into one mean-pooled embedding.

        Args:
            clip_path: Path to the clip mp4.

        Returns:
            A ``float32`` vector of shape ``[dim]``.
        """
        frames = self._read_frames(clip_path)
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0)  # mean-pool tokens -> [dim]
        return emb.float().cpu().numpy()
