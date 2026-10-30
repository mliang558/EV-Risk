#!/usr/bin/env python3
"""
Step 2: Build per-state charging networks from Step 1 station table (default 200 m cluster).

Method
------
1. Fixed-radius spatial clustering using WGS84 geodesic distance
   (sum capacity within each cluster).
2. Voronoi tessellation in a local azimuthal-equidistant projection
   (Delaunay fallback).
3. Edge weight = normalized WGS84 geodesic distance: d_ij / d_max.

Input (default)
---------------
  data/processed/stations_2026_48states.csv

Output (default)
----------------
  outputs/network_graph_2026_step2/network_structures/network_<STATE>.pkl
  outputs/network_graph_2026_step2/network_summary.csv

Each pickle: {"network": G, "meta": {...}}
  - node: lat, lon, location, n_l1, n_l2, n_dc, capacity, n_stations
    capacity = sum over cluster of (5*n_l1 + 25*n_l2 + 300*n_dc) per station
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
from pyproj import CRS, Geod, Transformer

from run_charging_network import STATES_ORDER

CLUSTER_RADIUS_M = 200.0
MIN_NODES = 3
CAPACITY_WEIGHTS = (5.0, 25.0, 300.0)  # L1, L2, DC
WGS84_GEOD = Geod(ellps="WGS84")


def hypernode_capacity(n_l1: float, n_l2: float, n_dc: float) -> float:
    w1, w2, wd = CAPACITY_WEIGHTS
    return w1 * n_l1 + w2 * n_l2 + wd * n_dc


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
    for col in ("n_l1", "n_l2", "n_dc"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0
    # Reconcile capacity with EVSE counts when Step-1 columns are present.
    cap_from_evse = df.apply(
        lambda r: hypernode_capacity(r["n_l1"], r["n_l2"], r["n_dc"]), axis=1
    )
    df["capacity"] = cap_from_evse
    return df.dropna(subset=["lat", "lon", "state"])


def cluster_stations(
    coords_ll: np.ndarray,
    capacities: np.ndarray,
    radius_m: float = CLUSTER_RADIUS_M,
    n_l1: np.ndarray | None = None,
    n_l2: np.ndarray | None = None,
    n_dc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic greedy clustering using WGS84 geodesic distance.

    Stations are sorted by latitude and longitude before clustering, so input
    row order cannot change the result.  Each cluster contains the first
    unassigned station and every still-unassigned station within ``radius_m``
    of that seed.
    """
    coords_ll = np.asarray(coords_ll, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    order = np.lexsort((coords_ll[:, 1], coords_ll[:, 0]))
    coords_ll = coords_ll[order]
    capacities = capacities[order]
    if n_l1 is not None:
        n_l1 = np.asarray(n_l1, dtype=np.float64)[order]
    if n_l2 is not None:
        n_l2 = np.asarray(n_l2, dtype=np.float64)[order]
    if n_dc is not None:
        n_dc = np.asarray(n_dc, dtype=np.float64)[order]

    used = np.zeros(len(coords_ll), dtype=bool)
    centers: list[np.ndarray] = []
    caps: list[float] = []
    counts: list[int] = []
    l1s: list[float] = []
    l2s: list[float] = []
    dcs: list[float] = []
    has_evse = n_l1 is not None and n_l2 is not None and n_dc is not None

    for i in range(len(coords_ll)):
        if used[i]:
            continue
        seed_lat, seed_lon = coords_ll[i]
        _, _, dists = WGS84_GEOD.inv(
            np.full(len(coords_ll), seed_lon),
            np.full(len(coords_ll), seed_lat),
            coords_ll[:, 1],
            coords_ll[:, 0],
        )
        dists = np.asarray(dists, dtype=np.float64)
        idx = (~used) & (dists <= radius_m)
        idx_indices = np.where(idx)[0]
        centers.append(coords_ll[idx_indices].mean(axis=0))
        counts.append(int(len(idx_indices)))
        if has_evse:
            s1 = float(n_l1[idx_indices].sum())
            s2 = float(n_l2[idx_indices].sum())
            sd = float(n_dc[idx_indices].sum())
            l1s.append(s1)
            l2s.append(s2)
            dcs.append(sd)
            caps.append(hypernode_capacity(s1, s2, sd))
        else:
            l1s.append(0.0)
            l2s.append(0.0)
            dcs.append(0.0)
            caps.append(float(capacities[idx_indices].sum()))
        used[idx_indices] = True

    return (
        np.asarray(centers),
        np.asarray(caps, dtype=float),
        np.asarray(counts, dtype=int),
        np.asarray(l1s, dtype=float),
        np.asarray(l2s, dtype=float),
        np.asarray(dcs, dtype=float),
    )


def _to_local_aeqd_meters(coords_ll: np.ndarray) -> tuple[np.ndarray, str]:
    """Project WGS84 points to a local azimuthal-equidistant CRS.

    The projection is centered on the analytical unit.  It has no fixed EPSG
    code; the complete PROJ definition is returned for provenance.
    """
    coords_ll = np.asarray(coords_ll, dtype=np.float64)
    lat_0 = float(np.mean(coords_ll[:, 0]))
    lon_0 = float(np.mean(coords_ll[:, 1]))
    proj4 = (
        f"+proj=aeqd +lat_0={lat_0:.12f} +lon_0={lon_0:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    target = CRS.from_proj4(proj4)
    transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    x, y = transformer.transform(coords_ll[:, 1], coords_ll[:, 0])
    return np.column_stack([x, y]).astype(np.float64), proj4


def _geodesic_distance_matrix(coords_ll: np.ndarray) -> np.ndarray:
    """Pairwise WGS84 geodesic distances in meters."""
    coords_ll = np.asarray(coords_ll, dtype=np.float64)
    n = len(coords_ll)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        _, _, dist = WGS84_GEOD.inv(
            np.full(n - i - 1, coords_ll[i, 1]),
            np.full(n - i - 1, coords_ll[i, 0]),
            coords_ll[i + 1 :, 1],
            coords_ll[i + 1 :, 0],
        )
        out[i, i + 1 :] = np.asarray(dist, dtype=np.float64)
        out[i + 1 :, i] = out[i, i + 1 :]
    return out


def add_hypernode_to_graph(
    G: nx.Graph,
    node_id: int,
    center: np.ndarray,
    capacity: float,
    n_stations: int,
    n_l1: float,
    n_l2: float,
    n_dc: float,
) -> None:
    G.add_node(
        node_id,
        lat=float(center[0]),
        lon=float(center[1]),
        location=(float(center[0]), float(center[1])),
        n_l1=float(n_l1),
        n_l2=float(n_l2),
        n_dc=float(n_dc),
        capacity=float(capacity),
        n_stations=int(n_stations),
    )


def voronoi_adjacency(coords_m: np.ndarray) -> set[tuple[int, int]]:
    try:
        vor = Voronoi(coords_m)
        adj: set[tuple[int, int]] = set()
        for p1, p2 in vor.ridge_points:
            a, b = (p1, p2) if p1 < p2 else (p2, p1)
            adj.add((a, b))
        return adj
    except Exception:
        # ``QJ`` resolves exactly collinear or nearly degenerate point sets
        # without changing the intended Delaunay fallback.
        tri = Delaunay(coords_m, qhull_options="QJ")
        adj = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                adj.add((a, b))
        return adj


def build_region_graph(
    df: pd.DataFrame,
    member_states: list[str] | tuple[str, ...],
    region_name: str,
    radius_m: float = CLUSTER_RADIUS_M,
    min_nodes: int = MIN_NODES,
) -> tuple[nx.Graph | None, dict | None]:
    """Build one graph after combining all station rows in an analytical unit."""
    sub = df[df["state"].isin(member_states)].copy()
    if len(sub) < 1:
        return None, None

    coords_ll = sub[["lat", "lon"]].values.astype(np.float64)
    capacities = sub["capacity"].values.astype(np.float64)
    n_l1 = sub["n_l1"].values.astype(np.float64)
    n_l2 = sub["n_l2"].values.astype(np.float64)
    n_dc = sub["n_dc"].values.astype(np.float64)
    n_raw = len(sub)

    centers, cluster_caps, cluster_counts, cl1, cl2, cdc = cluster_stations(
        coords_ll, capacities, radius_m=radius_m, n_l1=n_l1, n_l2=n_l2, n_dc=n_dc
    )
    n_nodes = len(centers)
    if n_nodes < min_nodes:
        return None, None

    coords_planar, voronoi_crs = _to_local_aeqd_meters(centers)
    if n_nodes < 4:
        adj = {(a, b) for a in range(n_nodes) for b in range(a + 1, n_nodes)}
        adjacency_method = "complete_small"
    else:
        adj = voronoi_adjacency(coords_planar)
        adjacency_method = "voronoi_local_aeqd"
    dist_matrix = _geodesic_distance_matrix(centers)
    dist_max = float(np.max(dist_matrix)) if n_nodes > 1 else 1.0
    if dist_max <= 0:
        dist_max = 1.0

    G = nx.Graph()
    for i in range(n_nodes):
        add_hypernode_to_graph(
            G,
            i,
            centers[i],
            float(cluster_caps[i]),
            int(cluster_counts[i]),
            float(cl1[i]),
            float(cl2[i]),
            float(cdc[i]),
        )

    for a, b in adj:
        d_m = float(dist_matrix[a, b])
        w = d_m / dist_max
        G.add_edge(a, b, weight=w, distance_m=d_m)

    if n_nodes > 0 and not nx.is_connected(G):
        raise RuntimeError(
            f"Network construction produced a disconnected graph for {region_name}: "
            f"{nx.number_connected_components(G)} components"
        )

    stats = {
        "state": region_name,
        "member_states": list(member_states),
        "n_raw": n_raw,
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "cluster_radius_m": float(radius_m),
        "dist_max_m": dist_max,
        "cluster_distance": "WGS84_geodesic",
        "station_order": "latitude_then_longitude",
        "adjacency": adjacency_method,
        "voronoi_crs": voronoi_crs,
    }
    return G, stats


def build_state_graph(
    df: pd.DataFrame,
    state: str,
    radius_m: float = CLUSTER_RADIUS_M,
) -> tuple[nx.Graph | None, dict | None]:
    """Backward-compatible wrapper for a single-state analytical unit."""
    return build_region_graph(df, [state], state, radius_m=radius_m, min_nodes=MIN_NODES)


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
                "node_features": "lat,lon,n_l1,n_l2,n_dc,capacity,n_stations",
                "capacity_formula": "5*n_l1+25*n_l2+300*n_dc",
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
