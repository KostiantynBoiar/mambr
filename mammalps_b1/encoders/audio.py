"""Frozen AST audio encoder: clip wav -> one pooled embedding.

Loads the clip's dedicated wav, resamples to mono at the configured rate, turns it into a
log-mel spectrogram (via the AST feature extractor), runs a frozen AST, and mean-pools the
token sequence into a single vector. Never trained (no gradients).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTModel

from mammalps_b1.config import Config
from mammalps_b1.utils.device import get_device


class AudioEncoder:
    """Frozen AST wrapper producing one embedding per clip."""

    def __init__(self, config: Config | None = None) -> None:
        """Load the frozen AST checkpoint and feature extractor.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()
        cfg = self.config.encoders
        self.sample_rate = cfg.audio_sample_rate
        self.device = get_device(cfg.device)
        self.extractor = ASTFeatureExtractor.from_pretrained(cfg.audio_ckpt)
        self.model = ASTModel.from_pretrained(cfg.audio_ckpt).to(self.device).eval()
        self.model.requires_grad_(False)

    @property
    def dim(self) -> int:
        """Embedding dimension (the backbone hidden size)."""
        return int(self.model.config.hidden_size)

    def _load_waveform(self, wav_path: Path | str) -> np.ndarray:
        """Load a wav as a mono 1-D array resampled to ``sample_rate``.

        Args:
            wav_path: Path to the clip wav.

        Returns:
            A ``float32`` 1-D waveform.
        """
        # soundfile reads wav directly (no ffmpeg/torchcodec backend needed).
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # [samples, channels]
        mono = wav.mean(axis=1)  # [samples]
        if sr != self.sample_rate:
            resampled = torchaudio.functional.resample(torch.from_numpy(mono), sr, self.sample_rate)
            mono = resampled.numpy()
        return mono.astype(np.float32)

    def embed(self, wav_path: Path | str) -> np.ndarray:
        """Encode a clip's audio into one mean-pooled embedding.

        Args:
            wav_path: Path to the clip wav.

        Returns:
            A ``float32`` vector of shape ``[dim]``.
        """
        waveform = self._load_waveform(wav_path)
        inputs = self.extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0)  # mean-pool tokens -> [dim]
        return emb.float().cpu().numpy()
