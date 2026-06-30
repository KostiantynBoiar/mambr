# Mask-Augmented Animal Behaviour Recognition

MSc dissertation, University of Glasgow. Does an animal's foreground silhouette **shape**
(extracted with SAM 2) improve behaviour recognition — especially for **posture/contact**
behaviours — where background/scene segmentation maps did not?

The project is organised as **two tracks**:

| | Track | Headline | Status |
|---|---|---|---|
| **Primary** | [`tracks/mabe22_contact/`](tracks/mabe22_contact/) | **Inter-animal mask-contact geometry** (mask IoU, contact-boundary length) vs the keypoint-hull geometry the field uses (SimBA/MARS), on **MABe22** mouse-triplets | scaffold; verify-first roadmap |
| **Secondary** | [`tracks/mammalps_baseline/`](tracks/mammalps_baseline/) | The **video / video+audio** baseline on **MammAlps** (single-animal), repurposed for shape-as-modality + background-robustness | done through the one-clip encoder probe |

> Re-scoped after a novelty/feasibility review: the defensible, novel core is *mask-contact
> geometry* (which needs multi-animal data → MABe22), so MammAlps became secondary. Shape itself
> is **not** claimed novel — it is established in human action/gait recognition; SAM 2 is the
> enabling technology that recovers a tracked foreground silhouette where the classical
> background-subtraction mechanism fails. **The pivot is pending the supervisor's sign-off.**

## Approach in one line
Frozen, off-the-shelf encoders (VideoMAE, an AST/AudioMAE-style audio model) **and SAM 2
inference-only** → features/masks **cached to disk once** → a **small trainable fusion head**.
The heavy models are never trained. Heavy caching runs on a Kaggle GPU; the light head
trains/evals on the M4 (PyTorch MPS).

## Layout
```
.claude/CLAUDE.md            master plan (both tracks)
common/                      shared code (extracted incrementally)
tracks/
  mabe22_contact/            PRIMARY — see its README + ROADMAP.md
  mammalps_baseline/         SECONDARY (the work so far) — see its README + ROADMAP.md
conftest.py                  puts the track packages on sys.path (tests + scripts)
requirements-local.txt(.lock), pyproject.toml   shared tooling; venv: mammalps-env/
```

## Run
```bash
mammalps-env/bin/python -m pytest -q          # data-free smoke + format-canary tests (from repo root)
# per-track usage: see each track's README (run scripts from inside the track dir)
```

Each track builds its own LaTeX reports (`report/{proposal,presentation}/`) with
`report/build.sh` (Tectonic + Ghostscript-normalise so PDFs preview on GitHub).
