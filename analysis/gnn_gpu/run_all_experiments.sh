#!/usr/bin/env bash
# One-shot: smoke test -> full pipeline (global CSV + train all models + eval + report)
#
# Usage (on server step4_gpu):
#   sed -i 's/\r$//' run_all_experiments.sh   # fix Windows CRLF if needed
#   chmod +x run_all_experiments.sh
#   nohup bash run_all_experiments.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH=.

PY="${PY:-$ROOT/.venv_gnn/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

DS="${DS:-pyg/pyg_dataset.pt}"
GCSV="${GCSV:-results_gnn/global_extended/window_global_features.csv}"
LGB_OUT="${LGB_OUT:-lgbm/train_lgbm_ext_br}"
BEST_CKPT="${BEST_CKPT:-pyg/train_gat_global_L5_ext_br/best_model.pt}"
LOO_EPOCHS="${LOO_EPOCHS:-40}"
GEO_EPOCHS="${GEO_EPOCHS:-40}"
SKIP_LOO="${SKIP_LOO:-1}"
SKIP_FULL_LOO="${SKIP_FULL_LOO:-1}"
SKIP_GEO="${SKIP_GEO:-0}"
SKIP_RADIUS="${SKIP_RADIUS:-0}"
HIDDEN="${HIDDEN:-64}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-20}"
SMOKE_LIMIT="${SMOKE_LIMIT:-40}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-3}"
NO_MLFLOW="${NO_MLFLOW:-1}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" results_gnn/global_extended pyg experiments lgbm
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/run_all_${TS}.log"

mlflow_args() {
  if [[ "$NO_MLFLOW" == "1" ]]; then
    echo --no-mlflow
  fi
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

run_py() {
  log "CMD: $PY -u $*"
  set +e
  "$PY" -u "$@" 2>&1 | tee -a "$MAIN_LOG"
  local st=${PIPESTATUS[0]}
  set -e
  if [[ "$st" -ne 0 ]]; then
    log "FAILED (exit $st): $*"
    exit "$st"
  fi
}

smoke_test() {
  log "======== SMOKE TEST (limit=${SMOKE_LIMIT}) ========"
  local smoke_csv="results_gnn/_smoke/window_global_features.csv"
  mkdir -p results_gnn/_smoke pyg/_smoke_train

  run_py step5_build_window_global_features.py --data-dir . \
    --feature-set extended --fast --limit "$SMOKE_LIMIT" \
    --out "$smoke_csv"

  run_py step6_train_gnn.py --dataset "$DS" --data-dir . \
    --model gat --num-layers 4 --hidden "$HIDDEN" \
    --global-features --global-feature-set extended \
    --global-features-csv "$smoke_csv" \
    --attacks no-capacity --limit "$SMOKE_LIMIT" \
    --epochs "$SMOKE_EPOCHS" --patience 2 \
    --out pyg/_smoke_train/gat_L4_ext \
    $(mlflow_args)

  run_py step7_evaluate_gnn.py --dataset "$DS" --data-dir . \
    --checkpoint pyg/_smoke_train/gat_L4_ext/best_model.pt \
    --global-features-csv "$smoke_csv" \
    --out results_gnn/_smoke_eval --no-plots \
    $(mlflow_args)

  log "SMOKE TEST OK"
}

train_one() {
  local name="$1"
  shift
  local out_dir="pyg/${name}"
  if [[ -f "${out_dir}/best_model.pt" && "${SKIP_EXISTING:-0}" == "1" ]]; then
    log "SKIP train (exists): ${out_dir}/best_model.pt"
    return 0
  fi
  log "======== TRAIN: $name ========"
  run_py step6_train_gnn.py --dataset "$DS" --data-dir . \
    --epochs "$EPOCHS" --patience "$PATIENCE" --hidden "$HIDDEN" \
    --global-features --global-feature-set extended \
    --global-features-csv "$GCSV" \
    --attacks no-capacity \
    --out "$out_dir" \
    $(mlflow_args) \
    "$@"
}

eval_one() {
  local name="$1"
  local ckpt="pyg/${name}/best_model.pt"
  local out="results_gnn/eval_${name}"
  log "======== EVAL: $name ========"
  run_py step7_evaluate_gnn.py --dataset "$DS" --data-dir . \
    --checkpoint "$ckpt" \
    --global-features-csv "$GCSV" \
    --out "$out" --no-plots \
    $(mlflow_args)
}

full_pipeline() {
  log "======== BUILD EXTENDED GLOBAL CSV (full, slow) ========"
  if [[ -f "$GCSV" && "${SKIP_GLOBAL_CSV:-0}" == "1" ]]; then
    log "SKIP global CSV (exists): $GCSV"
  else
    run_py step5_build_window_global_features.py --data-dir . \
      --feature-set extended --out "$GCSV"
  fi

  log "======== [9] LightGBM + Global extended 18 ========"
  run_py step7_window_lgb_baseline.py --data-dir . \
    --attacks no-capacity --feature-set global \
    --global-feature-set extended \
    --reuse-features \
    --out "$LGB_OUT"

  log "======== TRAIN GNN / MLP MODELS ========"
  train_one train_gat_global_L2_ext_br --model gat --num-layers 2
  train_one train_gat_global_L4_ext_br --model gat --num-layers 4
  train_one train_gat_global_L5_ext_br --model gat --num-layers 5
  train_one train_gt_global_L4_ext_br --model transformer --num-layers 4
  train_one train_gcn_global_L4_ext_br --model gcn --num-layers 4
  train_one train_mlp_global_ext_br --model mlp --num-layers 2
  train_one train_gat_global_L5_ext_betweenness --model gat --num-layers 5 --attack betweenness
  train_one train_gat_global_L5_ext_random --model gat --num-layers 5 --attack random

  log "======== EVAL ALL ========"
  for name in \
    train_gat_global_L2_ext_br \
    train_gat_global_L4_ext_br \
    train_gat_global_L5_ext_br \
    train_gt_global_L4_ext_br \
    train_gcn_global_L4_ext_br \
    train_mlp_global_ext_br \
    train_gat_global_L5_ext_betweenness \
    train_gat_global_L5_ext_random; do
    eval_one "$name"
  done

  log "======== ENHANCEMENT REPORT ========"
  run_py step7_build_enhancement_report.py --data-dir . \
    --manifest experiments/enhancement_manifest.json \
    --out results_gnn/enhancement_report

  run_generalization_studies

  log "======== DONE ========"
  log "Table: results_gnn/enhancement_report/enhancement_table.md"
  log "Log:  $MAIN_LOG"
}

find_best_ckpt() {
  for c in \
    "$BEST_CKPT" \
    pyg/train_gat_global_L5_ext_br/best_model.pt \
    pyg/train_gat_global_br/best_model.pt \
    pyg/train_gat_global/best_model.pt; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

run_generalization_studies() {
  local ckpt
  if ! ckpt="$(find_best_ckpt)"; then
    log "SKIP generalization: no best_model.pt under pyg/train_*"
    return 0
  fi
  log "Using checkpoint template: $ckpt"
  if [[ "$SKIP_GEO" != "1" ]]; then
    log "======== Geographic holdout: Northeast ========"
    run_py step7_leave_one_state_out.py --dataset "$DS" --data-dir . \
      --checkpoint "$ckpt" --global-features-csv "$GCSV" \
      --mode region --holdout-region northeast \
      --epochs "$GEO_EPOCHS" --out results_gnn/geo_holdout_northeast
    log "======== 5-state LOO ========"
    run_py step7_leave_one_state_out.py --dataset "$DS" --data-dir . \
      --checkpoint "$ckpt" --global-features-csv "$GCSV" \
      --mode loo --representative \
      --epochs "$LOO_EPOCHS" --out results_gnn/loo5_representative
  fi
  if [[ "$SKIP_FULL_LOO" != "1" && "$SKIP_LOO" != "1" ]]; then
    run_py step7_leave_one_state_out.py --dataset "$DS" --data-dir . \
      --checkpoint "$ckpt" --global-features-csv "$GCSV" \
      --mode loo --epochs "$LOO_EPOCHS" --out results_gnn/loo_full_48states
  fi
  if [[ "$SKIP_RADIUS" != "1" ]]; then
    run_py step7_r2_by_radius.py --dataset "$DS" --data-dir . \
      --checkpoint "$ckpt" --global-features-csv "$GCSV" \
      --windows-meta windows_meta.csv \
      --out results_gnn/r2_by_radius_L5_ext --no-plots
  fi
}

MODE="${1:-all}"
case "$MODE" in
  test-only) smoke_test ;;
  full-only) full_pipeline ;;
  all|*) smoke_test; full_pipeline ;;
esac
