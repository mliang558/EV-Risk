#!/usr/bin/env python3
"""
Compare metrics with removal-cache ON vs OFF (correctness smoke).

Usage (from Pro_directory):
  python analysis/attack_under_PO/counterfactual_batch1/smoke_compare_removal_cache.py \\
      --unit VT --n-sims 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def _run(unit: str, n_sims: int, out: Path, cache_on: bool, families: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}/analysis:{ROOT}/analysis/attack_under_PO:" + env.get(
        "PYTHONPATH", ""
    )
    env["CF_B1_REMOVAL_CACHE"] = "1" if cache_on else "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["FORCE"] = "1"
    cmd = [
        sys.executable,
        str(ROOT / "analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py"),
        "--only",
        unit,
        "--n-sims",
        str(n_sims),
        "--workers",
        "1",
        "--families",
        families,
        "--out",
        str(out),
        "--force",
        "--no-events",
    ]
    print(f"\n=== cache={'ON' if cache_on else 'OFF'} → {out} ===", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _load_all_metrics(unit_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(unit_dir.glob("metrics_*.csv")):
        df = pd.read_csv(p)
        df["__file"] = p.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics in {unit_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="VT")
    ap.add_argument("--n-sims", type=int, default=2)
    ap.add_argument("--families", default="baseline,CF-D,CF-S,U")
    ap.add_argument(
        "--out-root",
        default="results_cf_batch1_cache_compare",
        help="Under Pro_directory",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_on = out_root / "cache_on"
    out_off = out_root / "cache_off"
    for d in (out_on, out_off):
        if d.exists():
            shutil.rmtree(d)

    _run(args.unit, args.n_sims, out_on, cache_on=True, families=args.families)
    _run(args.unit, args.n_sims, out_off, cache_on=False, families=args.families)

    a = _load_all_metrics(out_on / args.unit)
    b = _load_all_metrics(out_off / args.unit)
    keys = ["sim_id", "scenario"]
    cols = [
        c
        for c in (
            "L_tilde",
            "L_event",
            "P_hit",
            "E_loss_given_hit",
            "n_events",
            "n_hit_events",
            "total_rel_loss_x_duration",
            "total_rel_loss_x_duration_hit",
        )
        if c in a.columns and c in b.columns
    ]
    m = a.merge(b, on=keys, suffixes=("_on", "_off"))
    bad = []
    for c in cols:
        d = (m[f"{c}_on"] - m[f"{c}_off"]).abs()
        # nan-safe
        both_nan = m[f"{c}_on"].isna() & m[f"{c}_off"].isna()
        ok = both_nan | (d.fillna(0) <= 1e-12)
        if not bool(ok.all()):
            bad.append((c, float(d[~both_nan].max()), int((~ok).sum())))
    if bad:
        print("FAIL: cache ON vs OFF mismatch:")
        for c, mx, n in bad:
            print(f"  {c}: max|Δ|={mx:.3e} n_bad={n}")
        sys.exit(1)
    print(f"PASS: {len(m)} metric rows identical (cache ON == OFF) for {args.unit}")
    man = out_on / args.unit / "manifest.json"
    if man.is_file():
        import json

        meta = json.loads(man.read_text())
        print(
            f"  removal_cache_entries={meta.get('removal_cache_entries')} "
            f"git={meta.get('git_commit')}"
        )


if __name__ == "__main__":
    main()
