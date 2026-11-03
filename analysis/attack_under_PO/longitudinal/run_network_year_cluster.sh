#!/usr/bin/env bash
# One longitudinal network year × pooled 2018--2023 outage model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

MODE="${1:-full}"
YEAR="${YEAR:?Set YEAR to one of 2018, 2019, 2020, 2021, or 2022}"
case "$YEAR" in
  2018|2019|2020|2021|2022) ;;
  *) echo "YEAR must be 2018--2022; 2023 is the canonical run" >&2; exit 2 ;;
esac

export ATTACK_RUN_LABEL="LONGITUDINAL-${YEAR}"
NETWORK_ROOT="${NETWORK_ROOT:-outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023}"
N_SIMS="${N_SIMS:-100}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
OUT="${OUT:-$ROOT/results_longitudinal_${RUN_DATE}_network${YEAR}_pooled2018_2023_sumlambda_n${N_SIMS}}"
VISIBLE_CPUS="$(nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$VISIBLE_CPUS}"

check_inputs() {
  python analysis/attack_under_PO/canonical/validate_inputs.py \
    --network-root "$NETWORK_ROOT" --year "$YEAR"
}

run_attack() {
  python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
    --network-root "$NETWORK_ROOT" \
    --network-year "$YEAR" \
    --out "$OUT" \
    --n-sims "$N_SIMS" \
    --workers "$WORKERS" \
    --families baseline \
    --epicenter-mode population \
    --no-events
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
