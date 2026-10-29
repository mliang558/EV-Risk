#!/usr/bin/env bash
# Formal manuscript run: 2023 networks × pooled 2018--2023 outage model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ATTACK_RUN_LABEL="${ATTACK_RUN_LABEL:-CANONICAL}"

MODE="${1:-run}"
NETWORK_ROOT="${NETWORK_ROOT:-outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023}"
N_SIMS="${N_SIMS:-100}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
OUT="${OUT:-$ROOT/results_canonical_${RUN_DATE}_2023_pooled_sumlambda_n${N_SIMS}}"
VISIBLE_CPUS="$(nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$VISIBLE_CPUS}"
ONLY="${ONLY:-}"

check_inputs() {
  python analysis/attack_under_PO/canonical/validate_inputs.py \
    --network-root "$NETWORK_ROOT"
}

run_attack() {
  local args=(
    --network-root "$NETWORK_ROOT"
    --out "$OUT"
    --n-sims "$N_SIMS"
    --workers "$WORKERS"
    --families baseline
    --epicenter-mode population
    --no-events
  )
  if [[ -n "$ONLY" ]]; then
    args+=(--only "$ONLY")
  fi
  if [[ "${FORCE:-0}" == "1" ]]; then
    args+=(--force)
  fi
  python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py "${args[@]}"
}

validate_outputs() {
  python analysis/attack_under_PO/canonical/validate_outputs.py \
    --out "$OUT" --expected-sims "$N_SIMS"
}

case "$MODE" in
  check) check_inputs ;;
  run) check_inputs; run_attack ;;
  validate) validate_outputs ;;
  full) check_inputs; run_attack; validate_outputs ;;
  *) echo "Usage: $0 {check|run|validate|full}" >&2; exit 2 ;;
esac
