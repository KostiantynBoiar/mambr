# tracks/mammalps_baseline — SECONDARY track

**What we've already built**, now re-scoped to *secondary*. A frozen-feature **video** and
**video+audio** behaviour-recognition baseline on **MammAlps Benchmark I**, scored with the
authors' official `eval_b1.py`. Pursued only **after** the primary MABe22 track lands; then
repurposed for (1) single-animal **shape-as-modality** and (2) **background-robustness** (RQ4).

- **Plan + status:** `ROADMAP.md` (S0–S9; done through the one-clip encoder probe).
- **Master spec:** `../../.claude/CLAUDE.md`.

```
mammalps_b1/   the package (config, data, encoders, models, scripts, utils, train/infer, run_eval.sh)
evaluation/    VENDORED official scorer (eval_b1.py + labels), pinned commit
kaggle/        Part A (Kaggle GPU) feature-cache setup
tests/         data-free smoke test + format-canary invariants
report/        proposal, dissertation, presentation, progress (this track's docs)
data/          dataset (gitignored)
ROADMAP.md
```

## Run (from inside this track)
```bash
cd tracks/mammalps_baseline
PYTHONPATH=. ../../mammalps-env/bin/python -m mammalps_b1.scripts.fetch_metadata   # ~660 KB
PYTHONPATH=. ../../mammalps-env/bin/python -m mammalps_b1.scripts.check_split      # gate: test==1244
bash mammalps_b1/run_eval.sh <results.txt>
```
Tests run from the **repo root** (`conftest.py` puts this track on the path):
`mammalps-env/bin/python -m pytest -q`.
