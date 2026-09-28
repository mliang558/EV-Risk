#!/usr/bin/env bash
# Batch 2 — year-specific A/B/C (baseline only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

OUT="${OUT:-$ROOT/results_batch2_yearly}"
N_SIMS="${N_SIMS:-100}"
NPROC_ALL="$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$NPROC_ALL}"
FAMILIES="${FAMILIES:-A B C}"
YEARS="${YEARS:-2018 2019 2020 2021 2022 2023}"
ONLY="${ONLY:-}"
NO_EVENTS="${NO_EVENTS:-1}"

mkdir -p "$OUT"
echo "[Batch2] OUT=$OUT N_SIMS=$N_SIMS WORKERS=$WORKERS FAMILIES=$FAMILIES YEARS=$YEARS"

ARGS=(
  --out "$OUT"
  --n-sims "$N_SIMS"
  --workers "$WORKERS"
  --families $FAMILIES
  --years $YEARS
)
if [[ -n "$ONLY" ]]; then
  ARGS+=(--only "$ONLY")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "$NO_EVENTS" != "1" ]]; then
  ARGS+=(--write-events)
fi

python analysis/attack_under_PO/batch2_yearly/run_parallel.py "${ARGS[@]}"
echo "[Batch2] finished -> $OUT"
