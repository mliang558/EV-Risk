#!/usr/bin/env python3
"""
Build EV charging networks (per state) and save only network pickle files.

This reuses the clustering + edge construction logic from `run_charging_network.py`,
but DOES NOT run any attack simulations. It is intended to prepare lightweight
network pickle files for downstream outage-attack analysis.

Usage (Windows PowerShell example)
----------------------------------

cd "C:/Users/maple/Desktop/EV_Project/Pro_directory"

python "analysis/build_network_pickle_only.py" `
  "data/raw/alt_fuel_stations_historical_day (Jan 1 2018).csv" `
  --out "outputs/network_pickle_2018"

This will create, for each state, a file:
  outputs/network_pickle_2018/run__<csv_basename>/network_structures/network_<STATE>.pkl

Each pickle contains:
  {"network": G}
where G is the NetworkX graph with:
  - node attributes: lat, lon, location=(lat, lon), capacity
  - edge attribute: weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from run_charging_network import (
    build_state_network,
    load_raw_csv,
    STATES_ORDER,
)


def build_pickle_for_csv(csv_path: Path, out_base: Path, verbose: bool = True) -> None:
    """Build per-state network pickle files for a single AFDC CSV."""
    df = load_raw_csv(csv_path)

    ts = csv_path.name.replace(" ", "_").replace(".", "_")
    run_dir = out_base / f"run__{ts}"
    net_dir = run_dir / "network_structures"
    net_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n=== Building networks for: {csv_path.name} ===")
        print(f"Output directory: {run_dir}")

    for idx, state in enumerate(STATES_ORDER):
        if verbose:
            print(f"\n[{idx + 1}/{len(STATES_ORDER)}] {state}")

        G, stats = build_state_network(df, state)
        if G is None:
            if verbose:
                print("  Skip (too few stations or build failed)")
            continue

        if verbose and stats:
            print(f"  Raw stations: {stats['n_raw']}")
            print(f"  Clusters: {stats['n_clusters']}")
            print(f"  Nodes: {stats['n_nodes']}")
            print(f"  Edges: {G.number_of_edges()}")

        # Sanity: ensure required attributes exist
        any_node = next(iter(G.nodes))
        node_data = G.nodes[any_node]
        for attr in ("lat", "lon", "location"):
            if attr not in node_data:
                raise ValueError(f"Node attribute '{attr}' missing in state {state} graph.")

        # Save pickle: only wrap the network
        data = {"network": G}
        net_path = net_dir / f"network_{state}.pkl"
        import pickle

        with open(net_path, "wb") as f:
            pickle.dump(data, f)

        if verbose:
            print(f"  ✓ Saved network pickle: {net_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-state EV charging network pickles (no attacks)."
    )
    parser.add_argument(
        "csv",
        help="Single AFDC CSV path (e.g. data/raw/alt_fuel_stations_historical_day (Jan 1 2018).csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/network_pickle",
        help="Output base directory (default: outputs/network_pickle)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_base = project_root / args.out

    csv_path = (
        project_root / args.csv
        if not Path(args.csv).is_absolute()
        else Path(args.csv)
    )
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_base.mkdir(parents=True, exist_ok=True)
    build_pickle_for_csv(csv_path, out_base, verbose=True)


if __name__ == "__main__":
    main()

