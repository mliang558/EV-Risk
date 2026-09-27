#!/usr/bin/env bash
# Counterfactual Batch 1 — smoke test before N_SIMS=100.
# Runs TX then VT, then automated checks. Exit 1 if hard checks fail.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/analysis:${ROOT}/analysis/attack_under_PO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SMOKE_OUT="${SMOKE_OUT:-results_cf_batch1_smoke}"
FAMILIES="${FAMILIES:-baseline,CF-D,CF-S,U,K}"
N_SIMS="${N_SIMS:-2}"
WORKERS="${WORKERS:-1}"
EPICENTER_MODE="${EPICENTER_MODE:-population}"
FORCE="${FORCE:-1}"
SKIP_POP_CHECK="${SKIP_POP_CHECK:-0}"

echo "============================================================"
echo " CF-B1 SMOKE  out=$SMOKE_OUT  N_SIMS=$N_SIMS  families=$FAMILIES"
echo "============================================================"

# 0) Tract pop units must exist (merged MD+DE+DC / CT+RI / Dakotas included)
if [[ "$SKIP_POP_CHECK" != "1" ]]; then
  if [[ ! -f data/processed/pop_units_epicenter.gpkg ]]; then
    echo "[smoke] building pop_units_epicenter.gpkg ..."
    python analysis/attack_under_PO/build_pop_units_epicenter.py
  fi
  python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
    --check-pop-units-only --project-root "$ROOT"
fi

# 1) TX — main CRN / mono / P(hit) / K / NEVI
echo ""
echo ">>> TX smoke (N_SIMS=$N_SIMS)"
T0=$(date +%s)
ONLY=TX N_SIMS="$N_SIMS" WORKERS="$WORKERS" FAMILIES="$FAMILIES" \
  OUT="$SMOKE_OUT" EPICENTER_MODE="$EPICENTER_MODE" FORCE="$FORCE" \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
T1=$(date +%s)
TX_SEC=$((T1 - T0))
echo "[smoke] TX wall time: ${TX_SEC}s"

python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
  --batch-dir "$SMOKE_OUT" --unit TX \
  --densify-cache outputs/network_graph_cf_densify_2023 \
  --estimate-full --smoke-seconds "$TX_SEC" --smoke-n-sims "$N_SIMS" \
  --n-units 45 --n-sims-full 100 --workers "${FULL_WORKERS:-12}" \
  --n-scenarios-smoke 14

# 2) VT — fallback / NEVI shortfall path (small state)
echo ""
echo ">>> VT smoke (fallback path)"
ONLY=VT N_SIMS="$N_SIMS" WORKERS="$WORKERS" FAMILIES="$FAMILIES" \
  OUT="$SMOKE_OUT" EPICENTER_MODE="$EPICENTER_MODE" FORCE="$FORCE" \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
  --batch-dir "$SMOKE_OUT" --unit VT --expect-fallback \
  --densify-cache outputs/network_graph_cf_densify_2023

echo ""
echo "============================================================"
echo " SMOKE PASSED — safe to launch N_SIMS=100"
echo "============================================================"
echo "Full run:"
echo "  N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S,U,K \\"
echo "    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh"
echo ""
echo "If runtime too long, drop K+U first:"
echo "  N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S \\"
echo "    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh"
echo ""
echo "Appendix (same seeds = same N_SIMS; seed=sim_id):"
echo "  OUT=results_cf_batch1_2023_pooled_epicenter_station EPICENTER_MODE=station \\"
echo "  N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S \\"
echo "    bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh"
