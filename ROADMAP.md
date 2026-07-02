# ZoomTrack — roadmap (M0–M6)

Coarse-to-fine, tile-refined promptable segmentation-and-tracking for SA-FARI's **small-animal**
gap. Verify-first: one reviewable artifact + one go/no-go gate per stage. **M2 is the kill-switch.**
Job spec in `.claude/CLAUDE.md`; dataset schema in `SA-FARI_dataset_reference.md`.

```
M0 access+baseline ─► M1 core logic+tests ─► M2 TILING ABLATION (★ premise / kill-switch)
   ─► M3 learned tile-selector ─► M4 fusion at scale + tracking ─► M5 full eval + ablations ─► M6 write-up
```

The two novel algorithms (`zoomtrack/fusion.py`, `zoomtrack/tile_selector.py` + `selector/`) are
built and unit-tested **first** (M1).

---

## M0 — Access + baseline
- **Do:** acquire SA-FARI (media from the CXL bucket 6 fps `JPEGImages_6fps/` + annotation JSONs from
  Hugging Face; `scripts/fetch_data.py`); load the `_ext` schema (`data/dataset.py`); run the
  backbone (SAM 3, or SAM 2 if weights are gated) coarse-only on a subset; emit VEval JSON
  (`writer.py`) and score with veval (`eval.py`, `scripts/score.py`); **reproduce the small-animal
  gap** on the small-masklet subset.
- **Gate:** the small-vs-large gap is real and measurable on your split.

## M1 — Core logic + tests
- **Do:** implement `geometry`, `masks`, **`fusion`**, the heuristic `tile_selector`, `tracking`, and
  the `mock` backbone. Unit-test fusion on the four cases + order-independence
  (`tests/test_fusion.py`); wire the mock-backbone `Pipeline` path.
- **Gate:** unit tests pass; the synthetic end-to-end smoke test shows **coarse-only recall <
  coarse+tiling recall** (small animal recovered).

## M2 — Tiling ablation  ★ premise check / kill-switch
- **Do:** with the real backbone, run **coarse-only** vs **fixed-tiling** (zoom every tile, no learned
  selection) on the small subset.
- **Gate:** fixed tiling raises small-masklet pDetA/pHOTA over baseline. **If not, the premise fails
  — PIVOT** (e.g. to the association/occlusion runner-up) before investing in the learned selector.

## M3 — Learned tile-selector
- **Do:** build per-tile labels from cached coarse dets + GT (`train_selector.build_tile_labels`);
  train the head (`selector/head.py`, class-balanced BCE); compare heuristic vs learned at **equal
  tile budget**.
- **Gate:** learned selection matches fixed-tiling pHOTA at a materially smaller tile budget (the
  efficiency claim); report selector precision/recall for "tile hides a missed animal".

## M4 — Fusion at scale + tracking
- **Do:** confirm fusion removes duplicates without dropping true instances on real videos; run the
  frozen backbone once and cache per `(video, frame, tile)` (`utils/io.py`); link masklets
  (`tracking.py`).
- **Gate:** no track-count explosion; large-animal pHOTA not regressed.

## M5 — Full eval + ablations
- **Do:** small-masklet pHOTA vs baseline and vs tiling-only; overall pHOTA / cgF1 / TETA; runtime;
  multi-seed (mean ± CI) where a head is trained.
- **Gate:** the §3 success criterion is met, or the honest negative is documented.

## M6 — Write-up
- **Do:** method, the two novel steps, the ablation ladder, and the accuracy-vs-runtime trade-off
  (into `report/`).
