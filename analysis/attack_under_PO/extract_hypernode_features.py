#!/usr/bin/env python3
"""
Extract per-hypernode features from a state network pickle.

Features produced for each hypernode:
- capacity
- degree
- betweenness_centrality
- clustering_coefficient
- nearest_neighbor_distance_km
- neighbor_mean_capacity
- neighbor_mean_degree
- neighbor_capacity_gini
- state_n_nodes
- state_mean_capacity
- state_capacity_gini
- year

Notes
-----
- "Hypernode" here means the 200m-clustered node saved in `network_*.pkl`.
- Degree/clustering are unweighted graph measures.
- Betweenness is computed on weighted shortest paths using edge attribute `weight`.
- Nearest-neighbor distance is geographic distance to the nearest other hypernode.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from compute_impact_radius import STATE_NAME_TO_ABBR


ABBR_TO_STATE_NAME = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}


def load_network(data_path: Path) -> nx.Graph:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data["network"]


def parse_state_abbr_from_path(data_path: Path) -> str | None:
    m = re.search(r"network_([A-Z]{2})\.pkl$", data_path.name)
    return m.group(1) if m else None


def parse_year_from_path(data_path: Path) -> int | None:
    path_str = str(data_path)
    for pattern in (r"network_pickle_(\d{4})", r"Jan[_ ]1[_ ](\d{4})"):
        m = re.search(pattern, path_str)
        if m:
            return int(m.group(1))
    return None


def haversine_km(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = arr.sum()
    if total <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * total)) - (n + 1) / n)


def get_node_location(node_data: dict) -> tuple[float, float]:
    loc = node_data.get("location")
    if loc is not None and len(loc) == 2:
        return float(loc[0]), float(loc[1])
    if "lat" in node_data and "lon" in node_data:
        return float(node_data["lat"]), float(node_data["lon"])
    raise ValueError("Node is missing both 'location' and ('lat', 'lon').")


def compute_nearest_neighbor_distances(G: nx.Graph) -> dict:
    nodes = list(G.nodes())
    locs = {node: get_node_location(G.nodes[node]) for node in nodes}
    out = {}
    if len(nodes) < 2:
        return {node: np.nan for node in nodes}

    for i, node_i in enumerate(nodes):
        loc_i = locs[node_i]
        best = np.inf
        for j, node_j in enumerate(nodes):
            if i == j:
                continue
            d = haversine_km(loc_i, locs[node_j])
            if d < best:
                best = d
        out[node_i] = float(best)
    return out


def extract_features(
    G: nx.Graph,
    *,
    state_abbr: str | None = None,
    state_name: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    nodes = list(G.nodes())
    capacities = {node: float(G.nodes[node].get("capacity", np.nan)) for node in nodes}
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clustering = nx.clustering(G)
    nearest_neighbor = compute_nearest_neighbor_distances(G)

    state_n_nodes = G.number_of_nodes()
    state_mean_capacity = float(np.nanmean(list(capacities.values()))) if capacities else np.nan
    state_capacity_gini = gini(capacities.values())

    rows = []
    for node in nodes:
        node_data = G.nodes[node]
        lat, lon = get_node_location(node_data)
        neighbors = list(G.neighbors(node))
        neighbor_caps = [capacities[nbr] for nbr in neighbors]
        neighbor_degs = [degrees[nbr] for nbr in neighbors]

        rows.append(
            {
                "node_id": node,
                "state_abbr": state_abbr,
                "state_name": state_name,
                "year": year,
                "lat": lat,
                "lon": lon,
                "capacity": capacities[node],
                "degree": degrees[node],
                "betweenness_centrality": float(betweenness.get(node, np.nan)),
                "clustering_coefficient": float(clustering.get(node, np.nan)),
                "nearest_neighbor_distance_km": nearest_neighbor[node],
                "neighbor_mean_capacity": float(np.nanmean(neighbor_caps)) if neighbor_caps else np.nan,
                "neighbor_mean_degree": float(np.nanmean(neighbor_degs)) if neighbor_degs else np.nan,
                "neighbor_capacity_gini": gini(neighbor_caps),
                "state_n_nodes": state_n_nodes,
                "state_mean_capacity": state_mean_capacity,
                "state_capacity_gini": state_capacity_gini,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-hypernode features from a network pickle.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to state network pickle, e.g. outputs/.../network_NY.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. Default: alongside pickle with suffix _hypernode_features.csv",
    )
    parser.add_argument("--state-abbr", type=str, default=None, help="Optional state abbreviation override, e.g. NY")
    parser.add_argument("--year", type=int, default=None, help="Optional year override")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {data_path}")

    state_abbr = args.state_abbr or parse_state_abbr_from_path(data_path)
    state_name = ABBR_TO_STATE_NAME.get(state_abbr) if state_abbr else None
    year = args.year if args.year is not None else parse_year_from_path(data_path)

    G = load_network(data_path)
    features_df = extract_features(G, state_abbr=state_abbr, state_name=state_name, year=year)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = data_path.with_name(data_path.stem + "_hypernode_features.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)
    print(f"Saved hypernode features to: {out_path}")
    print(features_df.head())


if __name__ == "__main__":
    main()
