#!/usr/bin/env bash
# Pack per-sim metrics needed for CF-D denominator / P(hit) audit (cluster).
# Run from Pro_directory. Output: cf_batch1_metrics_FOR_AUDIT.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
SRC="${SRC:-results_cf_batch1_2023_pooled}"
OUT="${OUT:-artifacts_cf_batch1_metrics_audit}"
rm -rf "$OUT"
mkdir -p "$OUT"
export SRC OUT

python - <<'PY'
from pathlib import Path
import csv
import os

src = Path(os.environ.get("SRC", "results_cf_batch1_2023_pooled"))
out = Path(os.environ.get("OUT", "artifacts_cf_batch1_metrics_audit"))
keep = [
    "sim_id", "scenario", "family", "n_nodes", "L_tilde",
    "total_rel_loss_x_duration", "total_rel_loss_x_duration_hit",
    "P_hit", "E_loss_given_hit", "n_events", "n_hit_events", "L_event",
]
n_files = 0
for udir in sorted(p for p in src.iterdir() if p.is_dir()):
    if udir.name.startswith("analysis"):
        continue
    dest = out / udir.name
    dest.mkdir(parents=True, exist_ok=True)
    for m in udir.glob("metrics_*.csv"):
        rows = list(csv.DictReader(m.open(encoding="utf-8")))
        if not rows:
            continue
        cols = [c for c in keep if c in rows[0]]
        with (dest / m.name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        n_files += 1
print(f"wrote {n_files} metrics files under {out}")
PY

zip -r cf_batch1_metrics_FOR_AUDIT.zip "$OUT"
ls -lh cf_batch1_metrics_FOR_AUDIT.zip
echo "Download zip -> local artifacts/cf_batch1_2023_pooled/metrics_by_unit/"
