#!/usr/bin/env bash
# Use 2 GPUs in parallel (no code change to step6 — separate processes per GPU).
#
# Mode A — one split, models split across GPUs (full sweep, ~2x faster):
#   bash run_stratified_splits_train_2gpu.sh
#
# Mode B — two splits in parallel (MINIMAL, 2x faster end-to-end):
#   PARALLEL_SPLITS=1 MINIMAL=1 bash run_stratified_splits_train_2gpu.sh
#
# Env: same as run_stratified_splits_train.sh + GPU0_ID GPU1_ID (physical indices, default 0 and 1)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH=.

PY="${PY:-$ROOT/.venv_gnn/bin/python}"
DS="${DS:-pyg/pyg_dataset.pt}"
GCSV="${GCSV:-results_gnn/global_extended/window_global_features.csv}"
SPLITS="${SPLITS:-stratified_stratum}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-20}"
HIDDEN="${HIDDEN:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_GLOBAL_CSV="${SKIP_GLOBAL_CSV:-1}"
MINIMAL="${MINIMAL:-0}"
NO_MLFLOW="${NO_MLFLOW:-0}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-0}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" results_gnn/global_extended pyg lgbm results_gnn/stratified_eval
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/stratified_train_2gpu_${TS}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

mlflow_args() {
  if [[ "$NO_MLFLOW" == "1" ]]; then echo --no-mlflow; else echo --mlflow --mlflow-experiment ev-charging-stratified-train; fi
}

run_train_gnn() {
  local gpu_id="$1"
  local base_name="$2"
  local split="$3"
  shift 3
  local out_dir="pyg/${base_name}_${split}"
  if [[ -f "${out_dir}/best_model.pt" && "$SKIP_EXISTING" == "1" ]]; then
    log "[GPU${gpu_id}] SKIP train: ${out_dir}/best_model.pt"
    return 0
  fi
  log "[GPU${gpu_id}] TRAIN ${base_name} | split=${split}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" -u step6_train_gnn.py --dataset "$DS" --data-dir . \
    --split "$split" --device cuda \
    --epochs "$EPOCHS" --patience "$PATIENCE" --hidden "$HIDDEN" \
    --eval-batch-size "${EVAL_BATCH_SIZE:-32}" \
    --global-features --global-feature-set extended \
    --global-features-csv "$GCSV" \
    --attacks no-capacity \
    --out "$out_dir" \
    $(mlflow_args) \
    "$@" 2>&1 | tee -a "$MAIN_LOG"
}

run_eval_gnn() {
  local gpu_id="$1"
  local base_name="$2"
  local split="$3"
  local ckpt="pyg/${base_name}_${split}/best_model.pt"
  local out="results_gnn/stratified_eval/${split}/eval_${base_name}"
  [[ -f "$ckpt" ]] || { log "[GPU${gpu_id}] SKIP eval (no ckpt): $ckpt"; return 0; }
  if [[ -f "${out}/test_metrics.json" && "$SKIP_EXISTING" == "1" ]]; then
    log "[GPU${gpu_id}] SKIP eval: ${out}/test_metrics.json"
    return 0
  fi
  log "[GPU${gpu_id}] EVAL ${base_name} | split=${split}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" -u step7_evaluate_gnn.py --dataset "$DS" --data-dir . \
    --split "$split" --device cuda \
    --checkpoint "$ckpt" --global-features-csv "$GCSV" \
    --out "$out" --no-plots \
    $(mlflow_args) 2>&1 | tee -a "$MAIN_LOG"
}

run_lgb() {
  local split="$1"
  local out="lgbm/train_lgbm_ext_br_${split}"
  if [[ -f "${out}/lgb_report.json" && "$SKIP_EXISTING" == "1" ]]; then
    log "SKIP LGB: ${out}/lgb_report.json"
    return 0
  fi
  log "TRAIN LGB (CPU) | split=${split}"
  "$PY" -u step7_window_lgb_baseline.py --data-dir . \
    --split "$split" --attacks no-capacity --feature-set global \
    --global-feature-set extended --reuse-features --out "$out" \
    2>&1 | tee -a "$MAIN_LOG"
}

# --- Mode B: one split per GPU (best for MINIMAL=1 + 2 splits) ---
if [[ "$PARALLEL_SPLITS" == "1" ]]; then
  IFS=',' read -ra SPLIT_ARR <<< "${SPLITS:-stratified_stratum,stratified_state}"
  s0="$(echo "${SPLIT_ARR[0]}" | xargs)"
  s1="$(echo "${SPLIT_ARR[1]:-${SPLIT_ARR[0]}}" | xargs)"
  log "PARALLEL_SPLITS: GPU${GPU0_ID}=$s0 | GPU${GPU1_ID}=$s1 | MINIMAL=$MINIMAL"
  (
    export CUDA_VISIBLE_DEVICES="$GPU0_ID" DEVICE=cuda SKIP_EXISTING MINIMAL SPLITS="$s0" EPOCHS PATIENCE HIDDEN NO_MLFLOW SKIP_GLOBAL_CSV
    bash "$ROOT/run_stratified_splits_train.sh"
  ) &
  pid0=$!
  (
    export CUDA_VISIBLE_DEVICES="$GPU1_ID" DEVICE=cuda SKIP_EXISTING MINIMAL SPLITS="$s1" EPOCHS PATIENCE HIDDEN NO_MLFLOW SKIP_GLOBAL_CSV
    bash "$ROOT/run_stratified_splits_train.sh"
  ) &
  pid1=$!
  wait "$pid0" "$pid1"
  log "DONE (parallel splits)"
  exit 0
fi

# --- Mode A: one split, models on two GPUs ---
split="$(echo "$SPLITS" | cut -d, -f1 | xargs)"
log "2-GPU model parallel | split=$split | GPU${GPU0_ID} vs GPU${GPU1_ID}"

if [[ ! -f "$GCSV" || "$SKIP_GLOBAL_CSV" != "1" ]]; then
  log "Building global CSV..."
  "$PY" -u step5_build_window_global_features.py --data-dir . --feature-set extended --out "$GCSV" \
    2>&1 | tee -a "$MAIN_LOG"
fi

run_lgb "$split"

if [[ "$MINIMAL" == "1" ]]; then
  run_train_gnn "$GPU0_ID" train_gat_global_L5_ext_br "$split" --model gat --num-layers 5
  run_eval_gnn "$GPU0_ID" train_gat_global_L5_ext_br "$split"
  log "DONE (MINIMAL single GPU — LGB is CPU; only one GNN)"
  exit 0
fi

# GPU0 batch
(
  run_train_gnn "$GPU0_ID" train_gat_global_L2_ext_br "$split" --model gat --num-layers 2
  run_train_gnn "$GPU0_ID" train_gat_global_L4_ext_br "$split" --model gat --num-layers 4
  run_train_gnn "$GPU0_ID" train_gat_global_L5_ext_br "$split" --model gat --num-layers 5
  run_train_gnn "$GPU0_ID" train_gt_global_L4_ext_br "$split" --model transformer --num-layers 4
) &
pid0=$!

# GPU1 batch
(
  run_train_gnn "$GPU1_ID" train_gcn_global_L4_ext_br "$split" --model gcn --num-layers 4
  run_train_gnn "$GPU1_ID" train_mlp_global_ext_br "$split" --model mlp --num-layers 2
  run_train_gnn "$GPU1_ID" train_gat_global_L5_ext_betweenness "$split" --model gat --num-layers 5 --attack betweenness
  run_train_gnn "$GPU1_ID" train_gat_global_L5_ext_random "$split" --model gat --num-layers 5 --attack random
) &
pid1=$!

wait "$pid0" "$pid1"

for name in \
  train_gat_global_L2_ext_br train_gat_global_L4_ext_br train_gat_global_L5_ext_br train_gt_global_L4_ext_br \
  train_gcn_global_L4_ext_br train_mlp_global_ext_br train_gat_global_L5_ext_betweenness train_gat_global_L5_ext_random; do
  # eval on same GPU as train (either works; use GPU0)
  run_eval_gnn "$GPU0_ID" "$name" "$split"
done

log "DONE 2-GPU | log=$MAIN_LOG"
