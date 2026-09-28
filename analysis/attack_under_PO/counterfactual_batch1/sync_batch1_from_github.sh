#!/usr/bin/env bash
# Sync CF Batch1 code from GitHub (cluster has no git).
# Run from anywhere; writes under ROOT (default: cwd if it looks like Pro_directory).
set -euo pipefail

REPO="${REPO:-mliang558/EV-Risk}"
REF="${REF:-main}"  # or commit hash e.g. b57c05e
BASE="https://raw.githubusercontent.com/${REPO}/${REF}"

if [[ -n "${1:-}" ]]; then
  ROOT="$(cd "$1" && pwd)"
elif [[ -d analysis/attack_under_PO ]]; then
  ROOT="$(pwd)"
else
  ROOT="/opt/data_repo/mliang_work/Pro_directory"
fi
cd "$ROOT"
echo "[sync] ROOT=$ROOT  REF=$REF"

FILES=(
  analysis/build_10km_panel_2018_2026.py
  analysis/attack_under_PO/attack_with_bootstrapped_outages.py
  analysis/attack_under_PO/counterfactual_batch1/crn_engine.py
  analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
  analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py
  analysis/attack_under_PO/counterfactual_batch1/build_5km_networks_2023.py
  analysis/attack_under_PO/counterfactual_batch1/smoke_compare_removal_cache.py
  analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py
  analysis/attack_under_PO/counterfactual_batch1/README.md
  analysis/attack_under_PO/counterfactual_batch1/sync_batch1_from_github.sh
)

for f in "${FILES[@]}"; do
  mkdir -p "$(dirname "$f")"
  url="${BASE}/${f}"
  echo "  GET $f"
  curl -fsSL "$url" -o "$f.tmp"
  mv "$f.tmp" "$f"
done

# strip CRLF if downloaded from mixed sources
sed -i 's/\r$//' analysis/attack_under_PO/counterfactual_batch1/*.sh 2>/dev/null || true
chmod +x analysis/attack_under_PO/counterfactual_batch1/*.sh

echo "[sync] done. Check:"
ls -la analysis/build_10km_panel_2018_2026.py \
  analysis/attack_under_PO/counterfactual_batch1/smoke_compare_removal_cache.py \
  analysis/attack_under_PO/attack_with_bootstrapped_outages.py
