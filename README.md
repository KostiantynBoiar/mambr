# ZoomTrack

**Coarse-to-fine, tile-refined promptable segmentation-and-tracking for small animals.**

Run a cheap coarse pass on the downscaled frame → **zoom into the tiles that likely hide missed
animals** → re-segment them at full resolution → **fuse across scales** into consistent masklets →
track. The heavy segmentation backbone stays **frozen**; only a small **tile-selector** (and
optionally a fusion head) is trained.

## Why
On **SA-FARI**, small animals are the biggest accuracy gap — and it's a *detection* failure caused
by whole-frame downscaling (~+29.2 pHOTA large-over-small; pDetA ≈38.5 small vs 72.5 large). Zooming
into the right tiles recovers them.

## Approach
Frozen **SAM 3** (SAM 2 fallback; a mock connected-components segmenter for tests) →
per-`(video, frame, tile)` cache (**Part A**, GPU, run once) → cheap **select → fuse → track → eval**
(**Part B**, iterate on a laptop). The two novel steps are the **cross-scale fusion** link-rule
(`zoomtrack/fusion.py`) and the **learned tile-selector** (`zoomtrack/tile_selector.py` +
`zoomtrack/selector/`).

## Layout
```
zoomtrack/            # the package (import zoomtrack)
  config.py types.py geometry.py masks.py fusion.py tile_selector.py tracking.py
  pipeline.py writer.py eval.py train_selector.py
  backbone/  (base · sam3 · sam2 · mock)      selector/ (features · head)
  data/dataset.py   utils/ (device · seed · io)   configs/*.yaml   scripts/*.py
tests/               # pure-logic unit tests (+ mock-backbone smoke)
report/build.sh      # LaTeX (Tectonic + Ghostscript)
data/ cache/ checkpoints/ third_party/   # gitignored
```

## Run
```bash
python3.11 -m venv zoomtrack-env
zoomtrack-env/bin/pip install -r requirements-local.txt          # Part B + mock backbone
zoomtrack-env/bin/python -m pytest -q                            # pure-logic tests; data/backbone tests skip
PYTHONPATH=. zoomtrack-env/bin/python -m zoomtrack.scripts.<name>  # fetch_data · run_baseline · run_pipeline · score
report/build.sh                                                 # build LaTeX docs
```
The Part-A caching box additionally needs `requirements-gpu.txt` (SAM 2/3 + veval).

## Evaluation
Predictions are written as SA-Co/VEval JSON and scored by the **official `veval`**
(`github.com/facebookresearch/sam3`) — the metric is never re-implemented. Headline number:
**small-masklet pHOTA** (with the pDetA/pAssA decomposition).

See `ROADMAP.md` (M0–M6, with the M2 kill-switch) and `.claude/CLAUDE.md` (the full job spec).
