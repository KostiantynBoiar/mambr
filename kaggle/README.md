# Kaggle GPU environment (Part A — feature extraction, run once)

Heavy, run-once feature caching happens here; light head training/eval happens on the M4
(see `requirements-local.txt`). The cache produced here is the hand-off to Part B.

## Setup
1. New Kaggle Notebook → **Settings → Accelerator: GPU**, **Internet: ON**.
2. First cell: paste `kaggle/00_setup_check.py`. It installs extras and runs the **env
   gate** (CUDA visible + frozen VideoMAE loads + one dummy clip embeds). Do not proceed
   until it prints `GATE PASSED`.

## Encoders (nearest-available HF, per the agreed decision)
- **Video:** `MCG-NJU/videomae-base` (ViT-B, Kinetics-pretrained, 16 frames @ 224²).
  Swap to the exact InternVideo condensed VideoMAE-K700 later if the authors share weights.
- **Audio:** default `MIT/ast-finetuned-audioset-10-10-0.4593` (AST, AudioSet) as an
  AudioMAE-adjacent, plug-and-play HF model — finalised during feature extraction.

## The real Kaggle risk: 87.9 GB data vs limited working disk
`mammalps_v1.zip` is ~87.9 GB; Kaggle's working dir is far smaller. Decide a staging
strategy **before** running the full extraction:
- **(a) Kaggle Dataset:** upload `mammalps_v1` (or a pre-extracted, video-only subset) as
  a private Kaggle Dataset and attach it read-only. Cleanest if it fits the size limits.
- **(b) Shard on the fly:** download a slice from Zenodo into `/kaggle/tmp`, extract +
  embed it, write the partial cache, delete raw, repeat. Survives the disk limit.
- **(c) Validate small first:** run the probe + extraction on a few hundred clips to confirm shapes and
  throughput, then scale to all 6,135.

## Output (hand-off to Part B)
`cache/<clip_id>.npz` (`video:[10,Dv]`, `audio:[10,Da]`, `window_meta`) +
`cache/manifest.parquet`. Download the `cache/` dir from Kaggle and point
`MAMMALPS_CACHE_ROOT` at it locally.
