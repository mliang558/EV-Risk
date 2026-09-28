#!/usr/bin/env bash
# Batch A/B — year-specific outage persistence. Separate from Batch 1.
#
# A: network year t × outage year t   (t = 2018..2023)
# B: network fixed 2023 × outage year t
#    B's t=2023 ≡ A's t=2023 → default B years = 2018..2022 only (5 runs)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DESIGN="${DESIGN:-A}"   # A | B
NETWORK_ROOT="${NETWORK_ROOT:-outputs/network_graph_10km_2018_2026}"
N_SIMS="${N_SIMS:-100}"
NPROC_ALL="$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$NPROC_ALL}"
ONLY="${ONLY:-}"
NO_EVENTS="${NO_EVENTS:-1}"
EPICENTER_MODE="${EPICENTER_MODE:-population}"

if [[ "$DESIGN" == "A" ]]; then
  OUT="${OUT:-$ROOT/results_cf_batchA_year_matched}"
  YEARS="${YEARS:-2018 2019 2020 2021 2022 2023}"
else
  OUT="${OUT:-$ROOT/results_cf_batchB_net2023_outage_year}"
  # Skip 2023: identical to Batch A 2023 (same net + same outage year)
  YEARS="${YEARS:-2018 2019 2020 2021 2022}"
fi
mkdir -p "$OUT"

echo "[CF-$DESIGN] ROOT=$ROOT OUT=$OUT"
echo "[CF-$DESIGN] N_SIMS=$N_SIMS WORKERS=$WORKERS YEARS=$YEARS NO_EVENTS=$NO_EVENTS EPICENTER=$EPICENTER_MODE"
echo "[CF-$DESIGN] threads OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"

ARGS=(
  --design "$DESIGN"
  --network-root "$NETWORK_ROOT"
  --out "$OUT"
  --n-sims "$N_SIMS"
  --workers "$WORKERS"
  --years $YEARS
  --epicenter-mode "$EPICENTER_MODE"
)
if [[ -n "$ONLY" ]]; then
  ARGS+=(--only "$ONLY")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "$NO_EVENTS" == "1" ]]; then
  ARGS+=(--no-events)
fi

python analysis/attack_under_PO/counterfactual_batch1/run_batch_ab_parallel.py "${ARGS[@]}"
echo "[CF-$DESIGN] done -> $OUT"
