"""One-clip frozen-encoder probe.

Proves the heavy path on a single real clip before extracting all 6,135: fetch the clip's
mp4 + wav (via HTTP range, like the split CSVs), run the frozen video and audio encoders, and
print the embedding shapes. Reveals the embedding dimensions the cache and fusion head need.

Usage::

    mammalps-env/bin/python -m mammalps_b1.scripts.probe_clip [--n 2] [--clip-id ID] [--config c.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from remotezip import RemoteZip

from mammalps_b1.config import Config
from mammalps_b1.data.splits import ClipRecord, SplitLoader
from mammalps_b1.encoders.audio import AudioEncoder
from mammalps_b1.encoders.video import VideoEncoder


class ClipProbe:
    """Fetch a clip's assets if needed and run both frozen encoders on it."""

    def __init__(self, config: Config | None = None) -> None:
        """Build the split loader and the frozen encoders.

        Args:
            config: Project config (defaults used when ``None``).
        """
        self.config = config or Config()
        self.loader = SplitLoader(self.config)
        self.video = VideoEncoder(self.config)
        self.audio = AudioEncoder(self.config)

    def _zip_member(self, subdir: str, suffix: str, rec: ClipRecord) -> str:
        """In-zip path for a clip asset, e.g. ``benchmark_1/clips/<dir>/<clip>.mp4``."""
        return f"{subdir}/{Path(rec.video_path).with_suffix(suffix)}"

    def _ensure_asset(self, member: str) -> Path:
        """Return the local path for an in-zip asset, fetching it via range if missing.

        Args:
            member: Path of the file inside the zip.

        Returns:
            The local path under ``data_root``.
        """
        local = self.config.paths.data_root / member
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            with RemoteZip(self.config.data.zip_url) as z:
                z.extract(member, path=self.config.paths.data_root)
            print(f"  fetched {member} ({local.stat().st_size:,} B)")
        return local

    def run(self, clip_id: str | None = None, n: int = 1) -> None:
        """Probe ``n`` test clips (or one specific ``clip_id``).

        Args:
            clip_id: Optional specific clip stem; otherwise the first ``n`` test clips.
            n: Number of clips to probe when ``clip_id`` is ``None``.

        Raises:
            AssertionError: If an embedding is non-finite or the wrong dimension.
            ValueError: If ``clip_id`` is given but not found in the test split.
        """
        records = self.loader.load("test")
        if clip_id is not None:
            records = [r for r in records if r.clip_id == clip_id]
            if not records:
                raise ValueError(f"clip_id {clip_id!r} not found in the test split")
        else:
            records = records[:n]

        print(
            f"device: {self.video.device} | video dim: {self.video.dim} | audio dim: {self.audio.dim}"
        )
        for rec in records:
            mp4 = self._ensure_asset(self._zip_member(self.config.data.clips_subdir, ".mp4", rec))
            wav = self._ensure_asset(self._zip_member(self.config.data.audios_subdir, ".wav", rec))
            v = self.video.embed(mp4)
            a = self.audio.embed(wav)
            assert v.shape == (self.video.dim,) and np.isfinite(v).all(), "bad video embedding"
            assert a.shape == (self.audio.dim,) and np.isfinite(a).all(), "bad audio embedding"
            print(
                f"{rec.clip_id} (dur {rec.end_s:.2f}s): "
                f"video shape={v.shape} finite=True | audio shape={a.shape} finite=True"
            )
        print("PROBE OK: real video + audio embeddings produced.")


def main() -> None:
    """Parse CLI args and run :class:`ClipProbe`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config overriding defaults")
    ap.add_argument("--clip-id", default=None, help="probe one specific clip stem")
    ap.add_argument("--n", type=int, default=1, help="number of test clips to probe")
    args = ap.parse_args()
    ClipProbe(Config.load(args.config)).run(clip_id=args.clip_id, n=args.n)


if __name__ == "__main__":
    main()
