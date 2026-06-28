#!/usr/bin/env bash
# Wrapper around the vendored official scorer.
# Usage: bash mammalps_b1/run_eval.sh <results.txt>
set -euo pipefail

RESULTS="${1:?usage: run_eval.sh <results.txt>}"
EVAL="evaluation/eval_b1.py"
LABELS="evaluation/labels_mapping_b1.json"
# Override with PYTHON=mammalps-env/bin/python to use the project venv.
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$EVAL" \
  --results_file_txt "$RESULTS" \
  --label_names_json "$LABELS" \
  --tasks Spe ActY ActN \
  --aggregate MEAN
