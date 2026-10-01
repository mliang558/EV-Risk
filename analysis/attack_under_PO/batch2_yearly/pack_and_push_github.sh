#!/usr/bin/env bash
# Pack Batch-2 results (metrics + analysis) and push to GitHub artifacts branch.
# Cluster Pro_directory is often NOT a git checkout — this clones a temp repo.
#
# Usage (on cluster):
#   cd /opt/data_repo/mliang_work/Pro_directory
#   bash analysis/attack_under_PO/batch2_yearly/pack_and_push_github.sh
#
# Env:
#   SRC          results dir (default: results_batch2_yearly)
#   REPO_URL     default: https://github.com/mliang558/EV-Risk.git
#   BRANCH       default: artifacts/cf-batch2-yearly
#   GH_TOKEN     optional; if set, uses https://x-access-token:${GH_TOKEN}@...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

SRC="${SRC:-results_batch2_yearly}"
REPO_URL="${REPO_URL:-https://github.com/mliang558/EV-Risk.git}"
BRANCH="${BRANCH:-artifacts/cf-batch2-yearly}"
STAGE="${STAGE:-artifacts_batch2_yearly_upload}"
ZIP_NAME="${ZIP_NAME:-batch2_yearly_metrics_analysis.zip}"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/metrics" "$STAGE/analysis_batch2"

echo "[1/4] Copy metrics_baseline.csv + summary.csv + r_st + manifests..."
# layout: family/tag/unit/metrics_baseline.csv
n_met=0
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  dest="$STAGE/metrics/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a "$f" "$dest"
  n_met=$((n_met + 1))
done < <(find "$SRC" -type f \( -name 'metrics_baseline.csv' -o -name 'summary.csv' -o -name 'manifest.json' \) -print0)

if [[ -f "$SRC/r_st_table.csv" ]]; then
  cp -a "$SRC/r_st_table.csv" "$STAGE/"
fi
if [[ -f "$SRC/batch_manifest.json" ]]; then
  cp -a "$SRC/batch_manifest.json" "$STAGE/"
fi
if [[ -d "$SRC/analysis_batch2" ]]; then
  cp -a "$SRC/analysis_batch2/." "$STAGE/analysis_batch2/" || true
fi

echo "  packed metrics/summary/manifest files: $n_met"
du -sh "$STAGE"

echo "[2/4] Zip..."
rm -f "$ZIP_NAME"
( cd "$STAGE" && zip -r -q "../$ZIP_NAME" . )
ls -lh "$ZIP_NAME"

echo "[3/4] Clone repo + artifacts branch..."
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

if [[ -n "${GH_TOKEN:-}" ]]; then
  CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/mliang558/EV-Risk.git"
else
  CLONE_URL="$REPO_URL"
fi

git clone --depth 1 "$CLONE_URL" "$TMP/repo"
cd "$TMP/repo"

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git fetch --depth 1 origin "$BRANCH"
  git checkout -B "$BRANCH" "origin/$BRANCH"
else
  git checkout --orphan "$BRANCH"
  git rm -rf . >/dev/null 2>&1 || true
  echo "# Batch2 yearly artifacts" > README.md
  git add README.md
  git commit -m "init artifacts branch for Batch2 yearly"
fi

mkdir -p artifacts/cf_batch2_yearly
rm -rf artifacts/cf_batch2_yearly/*
cp -a "$ROOT/$STAGE/." artifacts/cf_batch2_yearly/
cp -a "$ROOT/$ZIP_NAME" artifacts/cf_batch2_yearly/

# CSVs may be gitignored on main; force-add on artifacts branch
git add -f artifacts/cf_batch2_yearly
git status --short | head -40

echo "[4/4] Commit + push $BRANCH ..."
git commit -m "Add Batch2 yearly metrics + analysis_batch2 ($(date -u +%Y-%m-%d))" || {
  echo "Nothing new to commit?"
  exit 0
}
git push -u origin "$BRANCH"

echo
echo "DONE."
echo "  Branch: $BRANCH"
echo "  Path:   artifacts/cf_batch2_yearly/"
echo "  Zip:    artifacts/cf_batch2_yearly/$ZIP_NAME"
echo
echo "Local pull:"
echo "  git fetch origin $BRANCH"
echo "  git checkout origin/$BRANCH -- artifacts/cf_batch2_yearly"
