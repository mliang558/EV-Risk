#!/usr/bin/env bash
# Train + eval all enhancement models under stratified splits (NOT east_west reuse).
#
# Outputs (per split, e.g. stratified_stratum):
#   pyg/train_*_{split}/best_model.pt
#   lgbm/train_lgbm_ext_br_{split}/lgb_report.json
#   results_gnn/stratified_eval/{split}/eval_*/test_metrics.json
#
# Usage:
#   sed -i 's/\r$//' run_stratified_splits_train.sh
#   chmod +x run_stratified_splits_train.sh
#   # one split first (recommended):
#   SPLITS=stratified_stratum nohup bash run_stratified_splits_train.sh > logs/stratified_train.log 2>&1 &
#   # both splits:
#   nohup bash run_stratified_splits_train.sh > logs/stratified_train.log 2>&1 &
#
# Upload before first run (server copies may be old):
#   gnn/splits.py
#   step6_train_gnn.py          (--split)
#   step7_window_lgb_baseline.py (--split)
#   step7_evaluate_gnn.py       (--split)
#
# Env:
#   SPLITS=stratified_stratum,stratified_state
#   MINIMAL=1          # only LGB + GAT L5 (faster)
#   SKIP_EXISTING=1    # resume: skip steps whose outputs already exist (default ON)
#   CONTINUE_ON_ERROR=1  # keep going if one model fails
#   SKIP_GLOBAL_CSV=1  # if GCSV already built
#   EPOCHS=150 PATIENCE=20
#
# Progress: Step6 prints tqdm epoch bar (pip install tqdm). LGB is fast once CSV exists.
# Resume after kernel restart: re-run the same nohup command (SKIP_EXISTING=1).
#
# New checkpoints do NOT overwrite East/West:
#   pyg/train_gat_global_L5_ext_br/              <- old east_west
#   pyg/train_gat_global_L5_ext_br_stratified_stratum/  <- new split

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH=.

PY="${PY:-$ROOT/.venv_gnn/bin/python}"
DS="${DS:-pyg/pyg_dataset.pt}"
GCSV="${GCSV:-results_gnn/global_extended/window_global_features.csv}"
SPLITS="${SPLITS:-stratified_stratum,stratified_state}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-20}"
HIDDEN="${HIDDEN:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_GLOBAL_CSV="${SKIP_GLOBAL_CSV:-1}"
MINIMAL="${MINIMAL:-0}"
NO_MLFLOW="${NO_MLFLOW:-0}"
# Single-process GPU index: cuda:0, cuda:1, or cuda (default = GPU 0)
DEVICE="${DEVICE:-cuda}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" results_gnn/global_extended pyg lgbm results_gnn/stratified_eval
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/stratified_train_${TS}.log"

mlflow_args() {
  if [[ "$NO_MLFLOW" == "1" ]]; then
    echo --no-mlflow
  else
    echo --mlflow --mlflow-experiment ev-charging-stratified-train
  fi
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

preflight() {
  local missing=0
  for f in gnn/splits.py step6_train_gnn.py step7_window_lgb_baseline.py step7_evaluate_gnn.py; do
    if [[ ! -f "$f" ]]; then
      log "MISSING: $f"
      missing=1
    fi
  done
  if ! "$PY" step7_window_lgb_baseline.py -h 2>&1 | grep -q -- '--split'; then
    log "ERROR: step7_window_lgb_baseline.py on server lacks --split. Upload from analysis/gnn_gpu/"
    missing=1
  fi
  if ! "$PY" step6_train_gnn.py -h 2>&1 | grep -q -- '--split'; then
    log "ERROR: step6_train_gnn.py on server lacks --split. Upload from analysis/gnn_gpu/"
    missing=1
  fi
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
  log "Preflight OK (--split supported; SKIP_EXISTING=$SKIP_EXISTING)"
}

run_py() {
  log "CMD: $PY -u $*"
  set +e
  "$PY" -u "$@" 2>&1 | tee -a "$MAIN_LOG"
  local st=${PIPESTATUS[0]}
  set -e
  if [[ "$st" -ne 0 ]]; then
    log "FAILED (exit $st): $*"
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      return "$st"
    fi
    exit "$st"
  fi
}

preflight

if [[ ! -f "$GCSV" || "$SKIP_GLOBAL_CSV" != "1" ]]; then
  log "======== BUILD extended global CSV ========"
  run_py step5_build_window_global_features.py --data-dir . \
    --feature-set extended --out "$GCSV"
fi

train_gnn() {
  local base_name="$1"
  local split="$2"
  shift 2
  local out_dir="pyg/${base_name}_${split}"
  if [[ -f "${out_dir}/best_model.pt" && "$SKIP_EXISTING" == "1" ]]; then
    log "SKIP train (exists): ${out_dir}/best_model.pt"
    return 0
  fi
  log "======== TRAIN ${base_name} | split=${split} ========"
  run_py step6_train_gnn.py --dataset "$DS" --data-dir . \
    --split "$split" --device "$DEVICE" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --hidden "$HIDDEN" \
    --eval-batch-size "${EVAL_BATCH_SIZE:-32}" \
    --global-features --global-feature-set extended \
    --global-features-csv "$GCSV" \
    --attacks no-capacity \
    --out "$out_dir" \
    $(mlflow_args) \
    "$@"
}

eval_gnn() {
  local base_name="$1"
  local split="$2"
  local ckpt="pyg/${base_name}_${split}/best_model.pt"
  local out="results_gnn/stratified_eval/${split}/eval_${base_name}"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP eval (no ckpt): $ckpt"
    return 0
  fi
  if [[ -f "${out}/test_metrics.json" && "$SKIP_EXISTING" == "1" ]]; then
    log "SKIP eval (exists): ${out}/test_metrics.json"
    return 0
  fi
  log "======== EVAL ${base_name} | split=${split} ========"
  run_py step7_evaluate_gnn.py --dataset "$DS" --data-dir . \
    --split "$split" --device "$DEVICE" \
    --checkpoint "$ckpt" \
    --global-features-csv "$GCSV" \
    --out "$out" --no-plots \
    $(mlflow_args)
}

train_lgb() {
  local split="$1"
  local out="lgbm/train_lgbm_ext_br_${split}"
  if [[ -f "${out}/lgb_report.json" && "$SKIP_EXISTING" == "1" ]]; then
    log "SKIP LGB (exists): ${out}/lgb_report.json"
    return 0
  fi
  log "======== TRAIN LightGBM | split=${split} ========"
  run_py step7_window_lgb_baseline.py --data-dir . \
    --split "$split" \
    --attacks no-capacity --feature-set global \
    --global-feature-set extended \
    --reuse-features \
    --out "$out"
}

train_all_models() {
  local split="$1"
  train_lgb "$split"
  if [[ "$MINIMAL" == "1" ]]; then
    train_gnn train_gat_global_L5_ext_br "$split" --model gat --num-layers 5
    eval_gnn train_gat_global_L5_ext_br "$split"
    return 0
  fi
  train_gnn train_gat_global_L2_ext_br "$split" --model gat --num-layers 2
  train_gnn train_gat_global_L4_ext_br "$split" --model gat --num-layers 4
  train_gnn train_gat_global_L5_ext_br "$split" --model gat --num-layers 5
  train_gnn train_gt_global_L4_ext_br "$split" --model transformer --num-layers 4
  train_gnn train_gcn_global_L4_ext_br "$split" --model gcn --num-layers 4
  train_gnn train_mlp_global_ext_br "$split" --model mlp --num-layers 2
  train_gnn train_gat_global_L5_ext_betweenness "$split" --model gat --num-layers 5 --attack betweenness
  train_gnn train_gat_global_L5_ext_random "$split" --model gat --num-layers 5 --attack random

  for name in \
    train_gat_global_L2_ext_br \
    train_gat_global_L4_ext_br \
    train_gat_global_L5_ext_br \
    train_gt_global_L4_ext_br \
    train_gcn_global_L4_ext_br \
    train_mlp_global_ext_br \
    train_gat_global_L5_ext_betweenness \
    train_gat_global_L5_ext_random; do
    eval_gnn "$name" "$split"
  done
}

IFS=',' read -ra SPLIT_ARR <<< "$SPLITS"
for split in "${SPLIT_ARR[@]}"; do
  split="$(echo "$split" | xargs)"
  [[ -z "$split" ]] && continue
  log "########################################"
  log "SPLIT: $split"
  log "########################################"
  train_all_models "$split"
done

log "======== DONE ========"
log "Checkpoints: pyg/train_*_{split}/best_model.pt"
log "Metrics:     results_gnn/stratified_eval/{split}/eval_*/test_metrics.json"
log "LGB:         lgbm/train_lgbm_ext_br_{split}/lgb_report.json"
log "Log:         $MAIN_LOG"
