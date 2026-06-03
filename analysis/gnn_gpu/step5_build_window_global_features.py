#!/usr/bin/env python3
"""
Precompute window_global_features.csv (base 15-dim or extended 18-dim).

Extended adds (for betweenness-focused models):
  algebraic_connectivity, avg_shortest_path, degree_heterogeneity
  (diameter already in base 15-dim set)

Usage:
  cd step4_gpu
  PYTHONPATH=. python -u step5_build_window_global_features.py \\
    --data-dir . --feature-set extended \\
    --out results_gnn/global_extended/window_global_features.csv

  # Quick but missing Fiedler / exact paths (not for paper):
  PYTHONPATH=. python -u step5_build_window_global_features.py --feature-set extended --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.global_graph_features import (
    build_window_global_features_table,
    get_global_feature_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build window_global_features.csv")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument(
        "--feature-set",
        choices=["base", "extended"],
        default="extended",
        help="base=15 | extended=18 (+ Fiedler, avg path, degree heterogeneity)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip heavy NetworkX (extended topology will be approximate/zero)",
    )
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV (default: results_gnn/global_{set}/window_global_features.csv)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    meta = pd.read_csv(data_dir / args.windows_meta)
    wids = meta["window_id"].astype(str).tolist()
    if args.limit:
        wids = wids[: args.limit]

    out = Path(args.out) if args.out else (
        data_dir / "results_gnn" / f"global_{args.feature_set}" / "window_global_features.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    names = get_global_feature_names(args.feature_set)
    print(
        f"[build_global] windows={len(wids)} feature_set={args.feature_set} F={len(names)} fast={args.fast}",
        flush=True,
    )
    feat_df = build_window_global_features_table(
        data_dir, wids, feature_set=args.feature_set, fast=args.fast
    )
    feat_df.to_csv(out, index=False)

    meta_json = {
        "feature_set": args.feature_set,
        "n_features": len(names),
        "feature_names": list(names),
        "fast": args.fast,
        "n_windows": int(len(feat_df)),
        "out_csv": str(out),
    }
    with open(out.parent / "global_features_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    print(f"Saved -> {out}  ({len(feat_df)} rows × {len(names)} cols)", flush=True)
    if args.feature_set == "extended" and args.fast:
        print(
            "[warn] --fast: algebraic_connectivity / avg_shortest_path may be 0; "
            "re-run without --fast for full extended features.",
            flush=True,
        )


if __name__ == "__main__":
    main()
