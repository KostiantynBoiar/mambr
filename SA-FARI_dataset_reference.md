# SA-FARI — dataset reference (companion to the ZoomTrack spec)

> **Stub — the exact fields/paths below are CONFIRMED AT M0** by direct inspection
> (`zoomtrack/scripts/fetch_data.py` + `zoomtrack/data/dataset.py`). Treat unverified items as
> "to check", not fact.

## Downloads (two sources)
- **Media (frames):** the CXL Google bucket, aligned to the **6 fps** `JPEGImages_6fps/` frames.
  Inference runs at 6 fps to match the label timeline.
- **Annotations:** the JSON files from Hugging Face.

## Annotation schema (`_ext`)
A per-video record with **five top-level fields** (confirm names at M0), including:
- `file_names` — ordered list of the 6 fps frame paths (the label timeline).
- per-frame **RLE masklets** — frame-aligned to `file_names`; decode with `pycocotools`.
- `taxonomy` — species/taxon label (SAM 3's phrase prompt).
- `location_id` — camera/location identifier (enables cross-location / background-shift splits).
- negatives — `video_np_pairs` with `num_masklets == 0`: the model must return **nothing**; a false
  positive there is penalised by cgF1 (so selector *precision* matters, not just recall).

## Small-masklet subset (the headline)
Define "small" by the per-frame mask `areas` distribution — bucket at a percentile
(`config.eval.small_percentile`, default 0.25). The small subset is where small-masklet **pHOTA** is
reported.

## Scoring (VEval)
Emit predictions in the SA-Co/VEval JSON schema (`zoomtrack/writer.py`) and run the official
`veval` (`github.com/facebookresearch/sam3`) → `pHOTA` / `pDetA` / `pAssA` / `cgF1` / `TETA`. The
metric is never re-implemented here.

## Open items for M0
- [ ] Exact top-level field names + nesting of the `_ext` JSON.
- [ ] The split files (train/val/test) and how `location_id` groups them.
- [ ] First-frame boxes/points format (for the SAM 2 fallback prompt).
- [ ] The CXL bucket path + auth for the 6 fps media.
