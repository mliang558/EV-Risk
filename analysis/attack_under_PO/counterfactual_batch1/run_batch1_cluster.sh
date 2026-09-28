#!/usr/bin/env bash
# Counterfactual Batch 1 — cluster launcher (2023 × pooled outages, CRN).
# Does NOT touch results_mc_10km_panel_2018_2026/.
#
# Optimized defaults for a 64-core / high-RAM node:
#   - one process per analysis unit (WORKERS ≈ n_units)
#   - jobs submitted largest-|V| first (CA starts immediately)
#   - OMP/MKL threads = 1 inside each worker
#   - skip event dumps on full runs (metrics/summary still written)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

NETWORK_ROOT="${NETWORK_ROOT:-outputs/network_graph_10km_2018_2026}"
# Prefer absolute OUT so wipe path and run path never diverge
OUT="${OUT:-$ROOT/results_cf_batch1_2023_pooled}"
DENSIFY_CACHE="${DENSIFY_CACHE:-outputs/network_graph_cf_densify_2023}"
K5KM_ROOT="${K5KM_ROOT:-outputs/network_graph_5km_2018_2026}"
N_SIMS="${N_SIMS:-100}"
# Prefer all visible CPUs; runner caps at n_units
NPROC_ALL="$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$NPROC_ALL}"
# Include K (shares CRN with other families; 5 km nets built once if missing)
FAMILIES="${FAMILIES:-baseline,CF-D,CF-S,U,K}"
ONLY="${ONLY:-}"
EPICENTER_MODE="${EPICENTER_MODE:-population}"
# Full panel: skip per-event gz (huge I/O). Safe for §4.3:
# metrics_*.csv still stores n_events, n_hit_events, P_hit, L_event,
# E_loss_given_hit, total_rel_loss_x_duration, total_rel_loss_x_duration_hit.
# Smoke with event checks: NO_EVENTS=0
NO_EVENTS="${NO_EVENTS:-1}"

# Re-export so child Python sees them even if caller forgot
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$OUT" "$DENSIFY_CACHE"

echo "[CF-B1] ROOT=$ROOT"
echo "[CF-B1] OUT=$OUT N_SIMS=$N_SIMS WORKERS=$WORKERS FAMILIES=$FAMILIES EPICENTER=$EPICENTER_MODE NO_EVENTS=$NO_EVENTS"
echo "[CF-B1] threads OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"

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

python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py "${ARGS[@]}"

echo "[CF-B1] finished. Panel: ${OUT}/panel_scenario_summary.csv"
