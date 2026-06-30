# MammAlps baseline — SECONDARY track roadmap (S0–S9)

> **Re-scoped to secondary.** This was the original "Week 1" track; it is now the *secondary*
> track (see `../../.claude/CLAUDE.md`). **Do not rebuild it — it stays as-is.** It is pursued
> only **after** the primary MABe22 contact-geometry track lands (gated by the week-5 go/no-go).
> Its two narrowed jobs: (1) single-animal **shape-as-modality** ablation, and (2)
> **background-robustness** (the PanAf-FGBG inversion).

**Architecture:** frozen encoders (VideoMAE + AST) → embeddings cached once → a small trainable
fusion head. Heavy caching on a Kaggle GPU; light head training/eval on the M4 (MPS).

```
S0 setup ─► S1 data+split ─► S2 FORMAT CANARY ─► S3 one-clip encoder probe
   ─► S4 full feature cache (Kaggle) ─► S5 head overfit ─► S6 video-only ─► S7 +audio
   ─► S8 multi-seed ─► S9 lock baseline       ▲ key gate: video+audio > video
```

## Status (done so far)
- **S0–S3 done.** Env + vendored `eval_b1.py`; split verified (**train 4205 / val 686 / test
  1244 / total 6135**); format canary passes the 1244 assertion (~0.09 random avg mAP); frozen
  one-clip probe gives **768-d** video + audio embeddings on MPS.
- **S4 next when this track is resumed:** cache features for all 6,135 clips on Kaggle, then
  S5–S9 (train head; video, then video+audio; multi-seed; lock).

## Stage summaries (Start / Do / Output / Gate)
- **S0 setup** — vendored eval runs; class lists (Spe 5 / ActY 11 / ActN 19) load. ✅
- **S1 split** — parse the EPIC-style CSVs; build label vectors; **`len(test)==1244`**. ✅
- **S2 format canary** — `emit_results.py` writes a dummy results file (real labels, random preds,
  10 segments/clip); `eval_b1.py` runs past the 1244 assertion. ✅
- **S3 one-clip probe** — decode 16 frames → frozen VideoMAE; clip wav → frozen AST; finite 768-d
  embeddings. ✅
- **S4 full cache** — per clip, 10 windows, embed video+audio; `cache/<clip_id>.npz` +
  `manifest.parquet`; mask-ready layout. Gate: every clip cached, no NaNs.
- **S5 head overfit** — `CachedFeatureDataset` + summation-fusion head (Spe/ActY softmax, ActN
  sigmoid); overfit a tiny subset. Gate: loss → ~0.
- **S6 video-only** — train head; emit 10 rows/clip; score. Gate: real video-only mAP.
- **S7 +audio** — identical-capacity head + audio; **gate: video+audio > video-only.**
- **S8 multi-seed** — ≥3 seeds; mean ± std table.
- **S9 lock** — freeze baseline; confirm a `mask` modality slots in with no refactor.

## Re-scoped secondary jobs (once resumed)
1. **Shape-as-modality (RQ1/RQ2):** add a `mask_shape` stream to the mask-ready cache; ablation
   ladder **video → +mask → +audio → +audio+mask**; per-behaviour focus on chasing/courtship.
2. **Background-robustness (RQ4):** cross-camera / unseen-location `mask_shape` vs RGB degradation.

**Acceptance (unchanged):** clean eval past 1244 for video & video+audio, and **video+audio >
video-only**. Matching 0.453/0.473 exactly is a bonus (a frozen probe may land below the paper's
fine-tuned numbers — keep encoders frozen; flag the gap; don't unfreeze).

## Run (from inside this track)
```bash
cd tracks/mammalps_baseline
PYTHONPATH=. ../../mammalps-env/bin/python -m mammalps_b1.scripts.check_split
```
(Tests run from the repo root: `mammalps-env/bin/python -m pytest -q`.)
