#!/usr/bin/env bash
# Single entry point for the manuscript review experiments.
#
# The canonical state-ranking run is intentionally not repeated here. This script
# only runs (1) temporal robustness and (2) the optional intervention experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
N_SIMS="${N_SIMS:-100}"
WORKERS="${WORKERS:-45}"
REVIEW_ROOT="${REVIEW_ROOT:-$ROOT/results_review}"

export PYTHONPATH="$ROOT/analysis:$ROOT/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"

CF_DIR="$REVIEW_ROOT/cf_batch1_core"
B2_DIR="$REVIEW_ROOT/batch2_yearly"

check_environment() {
  python -c "import arviz, geopandas, networkx, xarray, netCDF4, shapely, scipy, pandas, numpy; print('environment OK')"
  python tests/test_dose_response_unique_doses.py
}

run_smoke() {
  check_environment
  FORCE=1 ONLY=TX,VT N_SIMS=2 WORKERS=1 NO_EVENTS=0 \
    FAMILIES=baseline,CF-D,CF-S OUT="$REVIEW_ROOT/smoke_cf_batch1" \
    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
  python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
    --batch-dir "$REVIEW_ROOT/smoke_cf_batch1" --unit TX
  python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
    --batch-dir "$REVIEW_ROOT/smoke_cf_batch1" --unit VT --expect-fallback

  FORCE=1 ONLY=VT,TX N_SIMS=2 WORKERS=1 FAMILIES="A B C" \
    OUT="$REVIEW_ROOT/smoke_batch2" \
    bash analysis/attack_under_PO/batch2_yearly/run_cluster.sh
  python analysis/attack_under_PO/batch2_yearly/smoke_test.py \
    --batch-dir "$REVIEW_ROOT/smoke_batch2" --units VT,TX
}

run_intervention() {
  N_SIMS="$N_SIMS" WORKERS="$WORKERS" NO_EVENTS=1 \
    FAMILIES=baseline,CF-D,CF-S OUT="$CF_DIR" \
    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
}

run_placement_controls() {
  local placement_dir="$REVIEW_ROOT/cf_batch1_placement_controls"
  N_SIMS="$N_SIMS" WORKERS="$WORKERS" NO_EVENTS=1 \
    FAMILIES=baseline,CF-D,CF-D-pop,CF-D-null,CF-S \
    OUT="$placement_dir" \
    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
  python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
    --batch-dir "$placement_dir"
  python analysis/attack_under_PO/counterfactual_batch1/audit_cf_d_denominator.py \
    --batch-dir "$placement_dir"
}

run_temporal() {
  N_SIMS="$N_SIMS" WORKERS="$WORKERS" NO_EVENTS=1 FAMILIES="A B C" \
    OUT="$B2_DIR" \
    bash analysis/attack_under_PO/batch2_yearly/run_cluster.sh
}

run_analysis() {
  python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
    --batch-dir "$CF_DIR"
  python analysis/attack_under_PO/counterfactual_batch1/audit_cf_d_denominator.py \
    --batch-dir "$CF_DIR"
  python analysis/attack_under_PO/batch2_yearly/analyze_batch2.py \
    --batch-dir "$B2_DIR"
  python analysis/attack_under_PO/batch2_yearly/check_mc_reliability.py \
    --batch-dir "$B2_DIR"
}

case "$MODE" in
  env) check_environment ;;
  smoke) run_smoke ;;
  intervention) run_intervention ;;
  placement) run_placement_controls ;;
  temporal) run_temporal ;;
  analyze) run_analysis ;;
  full)
    run_smoke
    run_intervention
    run_temporal
    run_analysis
    ;;
  *)
    echo "Usage: $0 {env|smoke|intervention|placement|temporal|analyze|full}" >&2
    exit 2
    ;;
esac
