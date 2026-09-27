#!/usr/bin/env bash
# Counterfactual Batch 1 — cluster launcher (2023 × pooled outages, CRN).
# Does NOT touch results_mc_10km_panel_2018_2026/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

NETWORK_ROOT="${NETWORK_ROOT:-outputs/network_graph_10km_2018_2026}"
OUT="${OUT:-results_cf_batch1_2023_pooled}"
DENSIFY_CACHE="${DENSIFY_CACHE:-outputs/network_graph_cf_densify_2023}"
K5KM_ROOT="${K5KM_ROOT:-outputs/network_graph_5km_2018_2026}"
N_SIMS="${N_SIMS:-100}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 8)}"
# Priority: policy claim first
FAMILIES="${FAMILIES:-baseline,CF-D,CF-S}"
ONLY="${ONLY:-}"

mkdir -p "$OUT" "$DENSIFY_CACHE"

echo "[CF-B1] ROOT=$ROOT"
echo "[CF-B1] OUT=$OUT N_SIMS=$N_SIMS WORKERS=$WORKERS FAMILIES=$FAMILIES"

# Optional: build 5 km nets if K is requested
if [[ "$FAMILIES" == *K* ]]; then
  if [[ ! -f "${K5KM_ROOT}/2023/network_summary.csv" ]]; then
    echo "[CF-B1] building 5 km networks for scenario K..."
    python analysis/attack_under_PO/counterfactual_batch1/build_5km_networks_2023.py \
      --out "$K5KM_ROOT"
  fi
fi

ARGS=(
  --network-root "$NETWORK_ROOT"
  --k5km-root "$K5KM_ROOT"
  --densify-cache "$DENSIFY_CACHE"
  --out "$OUT"
  --n-sims "$N_SIMS"
  --workers "$WORKERS"
  --families "$FAMILIES"
)
if [[ -n "$ONLY" ]]; then
  ARGS+=(--only "$ONLY")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py "${ARGS[@]}"

echo "[CF-B1] finished. Panel: ${OUT}/panel_scenario_summary.csv"
