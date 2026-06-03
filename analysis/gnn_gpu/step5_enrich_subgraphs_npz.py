#!/usr/bin/env python3
"""
Optional: bake lat, lon, dcfc_ratio into each subgraph .npz (faster Step 5 on GPU).

Requires windows_nodes.parquet + network_dir + stations CSV.

Usage:
  python step5_enrich_subgraphs_npz.py --data-dir data/step4_gpu \\
    --windows-nodes outputs/window_samples_2026_step3/windows_nodes.parquet \\
    --network-dir outputs/network_graph_2026_step2/network_structures \\
    --stations-csv data/processed/stations_2026_48states.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.node_features import build_dcfc_lookup, node_feature_matrix, window_graph_from_nodes

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich subgraph npz with node features")
    parser.add_argument("--data-dir", default="data/step4_gpu")
    parser.add_argument("--windows-nodes", required=True)
    parser.add_argument("--network-dir", required=True)
    parser.add_argument("--stations-csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest = pd.read_csv(data_dir / "manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)
    wnodes = pd.read_parquet(args.windows_nodes)
    dcfc = build_dcfc_lookup(Path(args.stations_csv))
    cache: dict = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Enrich npz"):
        wid = row["window_id"]
        sp = data_dir / row["subgraph_path"]
        wn = wnodes[wnodes["window_id"] == wid]
        G, old_keys = window_graph_from_nodes(wn, Path(args.network_dir), cache)
        x = node_feature_matrix(G, old_keys, dcfc)

        with np.load(sp, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        payload["lon"] = x[:, 0].astype(np.float32)
        payload["lat"] = x[:, 1].astype(np.float32)
        payload["dcfc_ratio"] = x[:, 3].astype(np.float32)
        np.savez_compressed(sp, **payload)

    print(f"Enriched {len(manifest)} files under {data_dir / 'subgraphs'}")


if __name__ == "__main__":
    main()
