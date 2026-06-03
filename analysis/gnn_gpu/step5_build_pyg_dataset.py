#!/usr/bin/env python3
"""
Step 5: Build PyG Data objects for window subgraphs.

Each graph:
  x: [lon, lat, capacity, dcfc_ratio]  (standardized globally)
  edge_index / edge_attr: Voronoi adjacency, normalized distance
  y: 30-dim vector = [betweenness×10, capacity×10, random×10]

Usage (GPU server, step4_gpu folder):
  python step5_build_pyg_dataset.py --data-dir . \\
    --windows-nodes ../windows_nodes.parquet \\
    --network-dir /path/to/network_graph_2026_step2/network_structures \\
    --stations-csv /path/to/stations_2026_48states.csv

If npz already contain lat/lon/dcfc_ratio arrays, network paths are optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.dataset import build_dataset, save_dataset_bundle
from gnn.splits import build_split_masks, format_stratum_report, split_key_prefix


def main() -> None:
    print("Step 5 starting (use: python -u step5_build_pyg_dataset.py ...)", flush=True)
    parser = argparse.ArgumentParser(description="Step 5: build PyG dataset")
    parser.add_argument("--data-dir", default="data/step4_gpu")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--windows-nodes", default=None, help="Step-3 windows_nodes.parquet")
    parser.add_argument("--network-dir", default=None, help="Step-2 network_*.pkl directory")
    parser.add_argument("--stations-csv", default=None, help="Step-1 CSV for DCFC ratio")
    parser.add_argument("--out", default=None, help="Output dir (default: data-dir/pyg)")
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument(
        "--splits",
        default="east_west,stratified_stratum,stratified_state",
        help="Comma-separated: east_west | stratified_stratum | stratified_state",
    )
    parser.add_argument("--test-frac", type=float, default=0.3, help="Test fraction for stratified splits")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU processes for graph build (try 8–16). Step 5 does not use GPU.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_dir = Path(args.out).resolve() if args.out else data_dir / "pyg"

    network_dir = Path(args.network_dir) if args.network_dir else None
    if network_dir is None:
        try:
            from network_paths import resolve_network_dir

            network_dir = resolve_network_dir(cwd=data_dir)
            print(f"[Step5] Auto network_dir={network_dir}", flush=True)
        except FileNotFoundError as e:
            print(f"[Step5] WARN: {e}", flush=True)

    wnodes = args.windows_nodes
    if wnodes is None:
        cand = data_dir / "windows_nodes.parquet"
        if cand.exists():
            wnodes = str(cand)

    graphs, meta_df, stats = build_dataset(
        data_dir,
        labels_path=Path(args.labels) if args.labels else None,
        windows_nodes_path=Path(wnodes) if wnodes else None,
        network_dir=network_dir,
        stations_csv=Path(args.stations_csv) if args.stations_csv else None,
        limit=args.limit,
        workers=args.workers,
    )

    include = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    manifest_path = data_dir / "manifest.csv"
    manifest_df = None
    if manifest_path.is_file():
        import pandas as pd

        manifest_df = pd.read_csv(manifest_path)
    split_masks = build_split_masks(
        meta_df,
        manifest=manifest_df,
        lon_threshold=args.lon_threshold,
        test_frac=args.test_frac,
        seed=args.split_seed,
        include=include,
    )

    path = save_dataset_bundle(out_dir, graphs, meta_df, stats, split_masks)
    print(f"Saved {len(graphs)} graphs -> {path}")
    for name in include:
        prefix = split_key_prefix(name)
        tr = split_masks[f"{prefix}_train"]
        te = split_masks[f"{prefix}_test"]
        print(f"\n=== Split: {prefix} ===")
        print(format_stratum_report(meta_df, tr, te))
    print(f"  Feature mean: {stats['feature_mean']}")
    print(f"  y shape per graph: {stats['y_dim']}")


if __name__ == "__main__":
    main()
