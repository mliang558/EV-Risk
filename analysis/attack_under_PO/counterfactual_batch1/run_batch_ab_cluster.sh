#!/usr/bin/env bash
# Batch A/B — year-specific outage persistence (#1). Separate from Batch 1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DESIGN="${DESIGN:-A}"   # A | B
NETWORK_ROOT="${NETWORK_ROOT:-outputs/network_graph_10km_2018_2026}"
N_SIMS="${N_SIMS:-100}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 8)}"
YEARS="${YEARS:-2018 2019 2020 2021 2022 2023}"
ONLY="${ONLY:-}"

if [[ "$DESIGN" == "A" ]]; then
  OUT="${OUT:-results_cf_batchA_year_matched}"
else
  OUT="${OUT:-results_cf_batchB_net2023_outage_year}"
fi
mkdir -p "$OUT"

ARGS=(
  --design "$DESIGN"
  --network-root "$NETWORK_ROOT"
  --out "$OUT"
  --n-sims "$N_SIMS"
  --workers "$WORKERS"
  --years $YEARS
)
if [[ -n "$ONLY" ]]; then
  ARGS+=(--only "$ONLY")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

python analysis/attack_under_PO/counterfactual_batch1/run_batch_ab_parallel.py "${ARGS[@]}"
echo "[CF-$DESIGN] done -> $OUT"
