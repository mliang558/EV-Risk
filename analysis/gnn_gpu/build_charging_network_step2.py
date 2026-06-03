#!/usr/bin/env python3
"""
Step 2: Build per-state charging networks from Step 1 station table (default 200 m cluster).

Method
------
1. 200 m radius spatial clustering (sum capacity within each cluster).
2. Voronoi tessellation adjacency (Delaunay fallback).
3. Edge weight = normalized geographic distance: d_ij / d_max (within state).

Input (default)
---------------
  data/processed/stations_2026_48states.csv

Output (default)
----------------
  outputs/network_graph_2026_step2/network_structures/network_<STATE>.pkl
  outputs/network_graph_2026_step2/network_summary.csv

Each pickle: {"network": G, "meta": {...}}
  - node: lat, lon, location, capacity, n_stations (raw sites in cluster)
  - edge: weight (= normalized distance), distance_m

Usage:
  python analysis/build_charging_network_step2.py
  python analysis/build_charging_network_step2.py --stations data/processed/stations_2025_48states.csv --out outputs/network_graph_2025_step2
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, Voronoi

from run_charging_network import STATES_ORDER, _to_epsg3857_meters

CLUSTER_RADIUS_M = 200.0
MIN_NODES = 3


def load_step1_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"state", "lat", "lon", "capacity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df.copy()
    df["state"] = df["state"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0)
    return df.dropna(subset=["lat", "lon", "state"])


def cluster_stations(
    coords_ll: np.ndarray,
    capacities: np.ndarray,
    radius_m: float = CLUSTER_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy fixed-radius clustering in EPSG:3857 meters."""
    coords_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])
    used = np.zeros(len(coords_m), dtype=bool)
    centers: list[np.ndarray] = []
    caps: list[float] = []
    counts: list[int] = []

    for i in range(len(coords_m)):
        if used[i]:
            continue
        dists = np.linalg.norm(coords_m - coords_m[i], axis=1)
        idx = (~used) & (dists <= radius_m)
        idx_indices = np.where(idx)[0]
        centers.append(coords_ll[idx_indices].mean(axis=0))
        caps.append(float(capacities[idx_indices].sum()))
        counts.append(int(len(idx_indices)))
        used[idx_indices] = True

    return np.asarray(centers), np.asarray(caps, dtype=float), np.asarray(counts, dtype=int)


def voronoi_adjacency(coords_3857: np.ndarray) -> set[tuple[int, int]]:
    try:
        vor = Voronoi(coords_3857)
        adj: set[tuple[int, int]] = set()
        for p1, p2 in vor.ridge_points:
            a, b = (p1, p2) if p1 < p2 else (p2, p1)
            adj.add((a, b))
        return adj
    except Exception:
        tri = Delaunay(coords_3857)
        adj = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                adj.add((a, b))
        return adj


def build_state_graph(
    df: pd.DataFrame,
    state: str,
    radius_m: float = CLUSTER_RADIUS_M,
) -> tuple[nx.Graph | None, dict | None]:
    sub = df[df["state"] == state]
    if len(sub) < MIN_NODES:
        return None, None

    coords_ll = sub[["lat", "lon"]].values.astype(np.float64)
    capacities = sub["capacity"].values.astype(np.float64)
    n_raw = len(sub)

    centers, cluster_caps, cluster_counts = cluster_stations(coords_ll, capacities, radius_m=radius_m)
    n_nodes = len(centers)
    if n_nodes < MIN_NODES:
        return None, None

    coords_3857 = _to_epsg3857_meters(centers[:, 0], centers[:, 1])
    adj = voronoi_adjacency(coords_3857)
    dist_matrix = np.linalg.norm(coords_3857[:, None] - coords_3857[None, :], axis=2)
    dist_max = float(np.max(dist_matrix))
    if dist_max <= 0:
        dist_max = 1.0

    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(
            i,
            lat=float(centers[i, 0]),
            lon=float(centers[i, 1]),
            location=(float(centers[i, 0]), float(centers[i, 1])),
            capacity=float(cluster_caps[i]),
            n_stations=int(cluster_counts[i]),
        )

    for a, b in adj:
        d_m = float(dist_matrix[a, b])
        w = d_m / dist_max
        G.add_edge(a, b, weight=w, distance_m=d_m)

    stats = {
        "state": state,
        "n_raw": n_raw,
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "cluster_radius_m": float(radius_m),
        "dist_max_m": dist_max,
    }
    return G, stats


def build_all(
    stations_path: Path,
    out_dir: Path,
    radius_m: float = CLUSTER_RADIUS_M,
    verbose: bool = True,
) -> pd.DataFrame:
    df = load_step1_stations(stations_path)
    net_dir = out_dir / "network_structures"
    net_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if verbose:
        print(f"Stations: {stations_path} ({len(df):,} rows)")
        print(f"Output:   {out_dir}")
        print(f"Cluster:  {radius_m:.0f} m | Edges: Voronoi | Weight: d_ij / d_max")

    for idx, state in enumerate(STATES_ORDER):
        G, stats = build_state_graph(df, state, radius_m=radius_m)
        if G is None:
            if verbose:
                print(f"[{idx + 1}/{len(STATES_ORDER)}] {state}: skip")
            continue

        payload = {
            "network": G,
            "meta": {
                **stats,
                "edge_weight": "normalized_distance",
                "adjacency": "voronoi",
            },
        }
        pkl_path = net_dir / f"network_{state}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        rows.append(stats)
        if verbose:
            print(
                f"[{idx + 1}/{len(STATES_ORDER)}] {state}: "
                f"raw={stats['n_raw']} nodes={stats['n_nodes']} edges={stats['n_edges']}"
            )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "network_summary.csv"
    summary.to_csv(summary_path, index=False)
    if verbose:
        print(f"\nSummary: {summary_path} ({len(summary)} states)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: 200m cluster + Voronoi + normalized distance")
    parser.add_argument(
        "--stations",
        type=str,
        default="data/processed/stations_2026_48states.csv",
        help="Step 1 CSV path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/network_graph_2026_step2",
        help="Output directory",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=CLUSTER_RADIUS_M,
        help="Cluster radius in meters (default: 200)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    radius_m = float(args.radius_m)
    stations_path = Path(args.stations)
    if not stations_path.is_absolute():
        stations_path = project_root / stations_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    if not stations_path.exists():
        print(f"File not found: {stations_path}", file=sys.stderr)
        sys.exit(1)

    build_all(stations_path, out_dir, radius_m=radius_m)


if __name__ == "__main__":
    main()
