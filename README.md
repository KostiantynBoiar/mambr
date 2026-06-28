# MammAlps Benchmark I — multimodal behaviour-recognition baseline

MSc dissertation (University of Glasgow). The first step of the project: a trustworthy
**video-only** and **video+audio** behaviour-recognition baseline on the
[MammAlps](https://github.com/eceo-epfl/MammAlps) Benchmark I dataset, scored with the
authors' official `eval_b1.py`. Later steps add a SAM2 foreground-silhouette **shape**
modality and test whether it improves recognition of posture/contact behaviours.

> This README is filled in as the project progresses.

## Approach in one line

Frozen encoders (VideoMAE for video, AudioMAE/AST-style for audio) → embeddings **cached
to disk once** → a **small trainable fusion head** on top. The encoders are never trained.
Heavy feature caching runs on a **Kaggle GPU** (Part A, once); the light head trains/evals
on the **M4 / MPS laptop** (Part B, repeatable). The head fuses modalities by summation so
its capacity is identical across `video` and `video+audio` — any gain reflects information,
not parameters — and a `mask` modality can be added later with no refactor.

## Repository structure

```
.
├── README.md                     # this file
├── evaluation/                   # VENDORED official scorer (do not edit), pinned commit
│   ├── eval_b1.py                #   the scoring contract — our predictions match this
│   ├── labels_mapping_b1.json    #   class spaces (ActY 11 · ActN 19 · Spe 5)
│   └── VENDORED_COMMIT.txt
├── mammalps_b1/                  # project source (OOP; all constants live in config.py)
│   ├── config.py                 # central Config: every constant w/ default + YAML/env override
│   ├── configs/                  # video.yaml, video_audio.yaml, config.example.yaml
│   ├── data/
│   │   ├── labels.py             # LabelSpace: class spaces + one-hot/multi-hot encoders
│   │   ├── splits.py             # SplitLoader + ClipRecord (parse CSVs -> label vectors)
│   │   └── dataset.py            # CachedFeatureDataset (modality-toggle)        (planned)
│   ├── encoders/                 # frozen VideoMAE / audio wrappers              (planned)
│   ├── models/
│   │   └── head.py               # MultiModalHead + MultiTaskLoss (config-driven)
│   ├── scripts/
│   │   ├── fetch_metadata.py     # MetadataFetcher: pull split CSVs via HTTP range
│   │   ├── check_split.py        # SplitChecker (verify test==1244, total==6135)
│   │   ├── emit_results.py       # format canary (dummy preds, real labels)
│   │   └── extract_features.py   # Part A: cache per-(clip,window) embeddings     (planned)
│   ├── utils/                    # util FUNCTIONS only (seed.py)
│   ├── train.py                  # Part B: train the head                        (planned)
│   ├── infer.py                  # Part B: emit results.txt (10 rows/clip)        (planned)
│   └── run_eval.sh               # wrapper around evaluation/eval_b1.py
├── kaggle/                       # Part A env: setup-check + data-staging notes
├── report/                       # proposal, dissertation, weekly progress (LaTeX)
├── tests/test_smoke.py          # data-free toolchain smoke test
├── requirements-local.txt       # Part B deps (+ .lock.txt frozen)
├── data/                         # dataset (gitignored)
└── cache/                        # cached features (gitignored)
```

## Dataset — MammAlps Benchmark I

- **Source:** Zenodo record [15040901](https://zenodo.org/records/15040901) —
`mammalps_v1.zip` (~87.9 GB) + `README.md`. A newer record
[15441330](https://zenodo.org/records/15441330) adds `dense_annotations_v1.zip`
(13.5 MB, per-frame dense annotations — not needed for B1 clip-level training).
- **6,135 single-animal clips**, split **train 4205 / val 686 / test 1244** (verified;
test == 1244 is what `eval_b1.py` asserts on).
- **Split CSVs** live *inside* the big zip at `benchmark_1/metadata/{train,val,test}.csv`.
They are small, so `mammalps_b1/scripts/fetch_metadata.py` pulls just those via HTTP
range requests — no 88 GB download needed to work on splits/labels.
- **CSV format** (comma-separated, with header):

  | column       | meaning                                                                                       |
  | ------------ | --------------------------------------------------------------------------------------------- |
  | `video_path` | `<track_dir>/<clip_id>.mp4`; `clip_id` = filename stem (e.g. `S1_C3_E154_V0066_ID1_T1_c0`)    |
  | `start_s`    | always `0.0` (clips are pre-cut)                                                              |
  | `end_s`      | clip duration in seconds (**ranges 0.03 → 96.6 s** — short clips need frame looping when features are extracted) |
  | `activity`   | one **ActY** class → one-hot[11]                                                              |
  | `actions`    | 1–2 **ActN** classes, `;`-separated, padded with `none` → **multi-hot[19]**                   |
  | `species`    | one **Spe** class → one-hot[5]                                                                |

- **Class spaces** (order fixed by `labels_mapping_b1.json`): ActY = 11 activities,
ActN = 19 actions, Spe = 5 species.
- **Heavy class imbalance** (test split): species ~98% `red_deer`; activities dominated by
`foraging` (688) and `vigilance` (228); the eventual contact/posture targets are rare
(`courtship` 56, `chasing` 3, `escaping` 1). Hence macro-mAP + class-balanced sampling.
- **In-zip layout (top level):** `benchmark_1/`, `benchmark_2/`, `raw_videos/`,
`reference_scenes/`, `events2files.json`, `file_list.txt`,
`raw_videos_mammalps_v1.csv`. The B1 clip mp4s' exact path is resolved during feature
extraction.

## Scoring contract (`eval_b1.py`)

Predictions are a tab-separated `.txt`, **one row per (test clip × sampled segment)**:

```
id \t sample_id \t test_segment_id \t ActY_preds \t ActN_preds \t Spe_preds \t ActY_label \t ActN_label \t Spe_label
```

- Column order is **fixed: preds then labels, each in order ActY, ActN, Spe** (regardless
of `--tasks` order). Each `*_preds`/`*_label` is a bracketed float vector `[a,b,c,...]`.
- `sample_id` = the clip's **row index in `test.csv`**; emit **~10 `test_segment_id` rows
per clip** — the scorer groups by `sample_id` and averages them (`--aggregate MEAN`),
then **asserts exactly 1244 rows**.
- **Spe & ActY** are multiclass (one-hot label, argmax). **ActN** is multi-label (multi-hot
label) → head uses sigmoid/BCE for ActN, softmax/CE for Spe & ActY.
- Metric: macro `average_precision_score` per task + an overall macro AP over the
concatenated 35-dim (11+19+5) vector (the headline "avg mAP", paper: **0.453** video /
**0.473** video+audio). See [evaluation/README.md](evaluation/README.md).

## Setup & usage

**Local (Part B — M4 / MPS), Python 3.11:**

```bash
python3.11 -m venv mammalps-env
mammalps-env/bin/pip install --upgrade pip
mammalps-env/bin/pip install -r requirements-local.txt
brew install ffmpeg
mammalps-env/bin/python -m pytest -q          # data-free smoke test
```

**Get the splits & verify:**

```bash
mammalps-env/bin/python -m mammalps_b1.scripts.fetch_metadata   # ~660 KB, not 88 GB
mammalps-env/bin/python -m mammalps_b1.scripts.check_split      # checks test==1244
```

**Kaggle (Part A — feature caching):** see [kaggle/README.md](kaggle/README.md).

**Score a results file:** `bash mammalps_b1/run_eval.sh <results.txt>`

## Status

**Done**

- Environment: repo, `mammalps-env` venv, vendored `eval_b1.py` at a pinned commit, Kaggle
  setup-check, data-free smoke test.
- Splits parsed + verified (train 4205 / val 686 / test 1244 / total 6135).
- Format canary: `emit_results.py` → `eval_b1.py` passes the 1244-clip check on dummy
  predictions (near-random avg mAP, confirming labels + aggregation are correct).

**Planned**

- One-clip frozen-encoder probe (VideoMAE / audio → first real embeddings).
- Full feature cache on Kaggle (per-clip, per-window embeddings).
- Train the head; video, then video+audio (target: video+audio > video).
- Multi-seed mean ± std report; lock the baseline.

## References

- Paper: Gabeff et al., *MammAlps* (CVPR 2025) — [https://arxiv.org/abs/2503.18223](https://arxiv.org/abs/2503.18223)
- Repo: [https://github.com/eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps) · Data: [https://zenodo.org/records/15040901](https://zenodo.org/records/15040901)
