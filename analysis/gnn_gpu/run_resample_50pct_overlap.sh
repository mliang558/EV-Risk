#!/usr/bin/env bash
# Resample 10k windows with max 50% disk overlap; cross-stratum dedupe; rebuild Step4–6.
#
# Run on step4_gpu (after uploading updated step3_*.py and gnn/window_overlap.py):
#   sed -i 's/\r$//' run_resample_50pct_overlap.sh
#   chmod +x run_resample_50pct_overlap.sh
#   nohup bash run_resample_50pct_overlap.sh > logs/resample_50pct.log 2>&1 &
#
# Env:
#   PY=./.venv_gnn/bin/python
#   NETWORK_DIR=./network_structures
#   SKIP_STEP4=1  # if subgraphs already match window ids (usually 0 after resample)

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results_sampling
export PYTHONPATH=.

PY="${PY:-./.venv_gnn/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found at $PY" >&2
  exit 1
fi
NETWORK_DIR="${NETWORK_DIR:-./network_structures}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/resample_50pct_${TS}.log"
mkdir -p logs results_sampling backup_windows_${TS}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Backup existing windows_meta / windows_nodes"
cp -a windows_meta.csv "backup_windows_${TS}/" 2>/dev/null || true
cp -a windows_nodes.parquet "backup_windows_${TS}/" 2>/dev/null || true

log "=== Step 3: random-seed sample + dedupe (max overlap 50%, cross-stratum) ==="
"$PY" -u step3_random_seed_sample.py \
  --network-dir "$NETWORK_DIR" \
  --out . \
  --target-total 10000 \
  --dedupe-centers \
  --max-overlap-frac 0.5 \
  --oversample-factor 5.0 \
  --dedupe-fill-rounds 10 \
  --placement-order radius_asc \
  --dedupe-strategy per_stratum \
  --seed 42 \
  2>&1 | tee -a "$LOG"

log "=== Verify overlap (pairwise audit) ==="
"$PY" -u step3_dedupe_window_centers.py \
  --meta windows_meta.csv \
  --nodes windows_nodes.parquet \
  --out windows_meta.csv \
  --nodes-out windows_nodes.parquet \
  --target 10000 \
  --max-overlap-frac 0.5 \
  --verify \
  --report "results_sampling/dedupe_report_${TS}.json" \
  2>&1 | tee -a "$LOG"

if [[ "${SKIP_STEP4:-0}" != "1" ]]; then
  log "=== Step 4: export subgraphs ==="
  "$PY" -u step4_export_window_subgraphs.py \
    --windows-dir . \
    --network-dir "$NETWORK_DIR" \
    --out . \
    2>&1 | tee -a "$LOG"

  log "=== Step 4: y labels ==="
  "$PY" -u step4_compute_y_labels.py \
    --data-dir . \
    --workers "${WORKERS:-16}" \
    2>&1 | tee -a "$LOG"
fi

log "=== Step 5: pyg dataset (all splits) ==="
"$PY" -u step5_build_pyg_dataset.py \
  --data-dir . \
  --splits east_west,stratified_stratum,stratified_state \
  --test-frac 0.3 --split-seed 42 \
  2>&1 | tee -a "$LOG"

log "=== Step 5: global features CSV (if missing) ==="
if [[ ! -f results_gnn/global_extended/window_global_features.csv ]]; then
  "$PY" -u step5_build_window_global_features.py \
    --data-dir . --feature-set extended \
    --out results_gnn/global_extended/window_global_features.csv \
    2>&1 | tee -a "$LOG"
fi

log "=== Retrain: stratified splits (see run_stratified_splits_train.sh) ==="
log "  MINIMAL=1 SPLITS=stratified_stratum bash run_stratified_splits_train.sh"
log "Done. Log: $LOG"
