# Vendored evaluation (do not edit)

`eval_b1.py` and `labels_mapping_b1.json` are copied **verbatim** from the official
[eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps) repository (MIT licence),
pinned at the commit in `VENDORED_COMMIT.txt`. They are the scoring contract for
Benchmark I and must not be modified — our predictions are formatted to match them.

Run via `mammalps_b1/run_eval.sh <results.txt>` or directly:

```
python evaluation/eval_b1.py \
  --results_file_txt <results.txt> \
  --label_names_json evaluation/labels_mapping_b1.json \
  --tasks Spe ActY ActN \
  --aggregate MEAN
```

Contract recap (confirmed against source): TSV rows `id \t sample_id \t test_segment_id`
then **preds in order ActY, ActN, Spe**, then **labels in order ActY, ActN, Spe**; each
pred/label is a bracketed float vector `[..]`. Rows are grouped by `sample_id`, aggregated
(MEAN/MAX) over `test_segment_id`, then **asserted to be exactly 1244**. Spe & ActY are
multiclass (one-hot label, argmax); ActN is multi-label (multi-hot). Metric is macro
`average_precision_score` per task plus an overall macro AP over the concatenated
35-dim vector.
