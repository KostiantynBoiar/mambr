# MABe22 contact-behaviour — label & eval contract (M1, LOCKED)

> The MABe22 analog of the MammAlps `eval_b1.py` contract + the 1244-clip canary. This is the
> **single, model-agnostic** evaluation the whole study reports against. Confirmed against the
> data at M0 (`mabe22/scripts/check_data.py`) and exercised end-to-end by the canary
> (`mabe22/scripts/emit_dummy.py`).

## Data (confirmed at M0)
The study uses the **submission set** — the one with video + keypoints + labels. Labels live in
`mouse_submission_labels.npy` (flat MABe format): `vocabulary` (13 tasks),
`label_array` `float32 [13, 3,294,000]` (0/1 for the Discrete behaviour tasks),
`frame_number_map` `{sequence_id: (start, end)}` over **1830 sequences × 1800 frames**, and
`task_type`. (The `mouse_triplet_train/test` files are the representation-learning challenge data
— no video, only chases+lights labelled — and are not used here.)

## Target behaviours (4, framewise, independent binary)
Exact vocabulary names: `chases`, `huddles`, `oral_oral_contact` (≈ oral contact),
`oral_genital_contact` (≈ oral–genital). The other tasks are conditions (strain, light cycle,
day, time) plus extra contact behaviours (`approaches`, `close`, `contact`, `oral_ear_contact`,
`watching`) available if wanted later.

- **Multi-label, not multi-class:** behaviours can co-occur; each is an independent binary label
  per frame, scored separately.
- **Dense but heavily imbalanced:** every frame carries a 0/1 for each behaviour (no missing
  values). Positives are rare — M0: **chases 0.05 %, oral-oral / oral-genital ≈ 0.23 %, huddles
  8.1 %** — so **average precision is the more informative metric for the rare behaviours** (F1 at
  a fixed threshold can be near-zero), and class-balanced sampling matters at train time.

## Unit of prediction
A **frame** belonging to a **sequence**. Each sample carries a 4-dim **score** vector and a 4-dim
0/1 **label** vector (one per behaviour, in `Config.tasks.behaviours` order). Labels are dense, so
no per-frame valid mask is needed.

## Split — leakage-safe, by sequence
- Group **all frames of a sequence** into one split. **Never** split by random frame/window
  (adjacent frames are near-identical → leakage).
- Fixed, seeded split of the 1830 sequences (`Config.eval.val_frac`/`test_frac`/`seed`;
  default 70/15/15 → ≈ 1282/274/274 sequences).
- **Invariant (asserted by the canary test):** sequence-id sets of any two splits are disjoint.

## Metric (model-agnostic)
Per behaviour, over all frames of the split: **F1** at a fixed threshold (MABe convention) **and**
**average precision** (threshold-free; comparability with the MammAlps secondary track). Report
each behaviour plus the **macro** average. The *model* that produced the scores (a linear/Ridge
probe for the clean mask-vs-hull comparison at M6, or the small fusion head at M9) is **not** part
of this contract.

## Format canary (verified)
`mabe22/scripts/emit_dummy.py` writes **random scores** with the **real labels** for the test
split; `MABeScorer` runs end-to-end and prints per-behaviour F1 + AP (near-random by design). This
is the MABe22 analog of MammAlps `emit_results.py` → `eval_b1.py`: it proves the eval plumbing is
correct before any geometry/model exists. ✅ ran clean (macro F1 ≈ 0.04, AP ≈ 0.02).

## What is NOT decided here (later stages)
The feature sets (`mask_contact`, `mask_shape`, `kp_hull`), the model, and the windowing /
subsampling — those are M4–M6. This contract only fixes *what counts as a correct answer and how
it is scored*.
