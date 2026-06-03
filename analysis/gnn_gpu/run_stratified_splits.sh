#!/usr/bin/env bash
# Add stratified-split results for ALL manifest models (does not touch east_west eval dirs).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PY:-./.venv_gnn/bin/python}"

echo "=== Stratified eval: all models (new dirs only) ==="
MLFLOW_URI="${MLFLOW_URI:-file:./mlruns}"
MLFLOW_ARGS=(--mlflow --mlflow-experiment ev-charging-stratified-splits --mlflow-tracking-uri "$MLFLOW_URI")
if [[ "${NO_MLFLOW:-0}" == "1" ]]; then
  MLFLOW_ARGS=(--no-mlflow)
fi

"$PY" -u step7_run_stratified_splits_all_models.py \
  --data-dir . \
  --manifest experiments/enhancement_manifest.json \
  --skip-existing \
  --device cuda \
  "${MLFLOW_ARGS[@]}"

echo "Done. See results_gnn/stratified_eval/*/comparison_table.md"
echo "MLflow UI: mlflow ui --backend-store-uri $MLFLOW_URI --host 0.0.0.0 --port 5000"
