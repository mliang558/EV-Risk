#!/usr/bin/env python3
"""
Build state-level charging networks (2018–2026) with 10 km hypernode clustering.

Per unit
--------
- Greedy 10 km clustering (EPSG:3857), Voronoi adjacency (Delaunay fallback)
- Edge weight w_ij = d_ij / d_max

Fixed merges (all years)
------------------------
- Maryland  : DC + DE + MD  -> network_Maryland.pkl
- Dakotas   : SD + ND       -> network_Dakotas.pkl
- Connecticut : RI + CT     -> network_Connecticut.pkl

Output (default)
----------------
  outputs/network_graph_10km_2018_2026/
    <year>/network_structures/network_<unit>.pkl
    <year>/network_summary.csv
    panel_hypernode_counts.csv
    README.txt

Usage
-----
  cd Pro_directory
  python analysis/build_10km_panel_2018_2026.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import build_charging_network_step2 as s2  # noqa: E402
from build_charging_network_step2 import (  # noqa: E402
    add_hypernode_to_graph,
    load_step1_stations,
)
from prepare_stations import default_csv_path, prepare_stations  # noqa: E402
from run_charging_network import STATES_ORDER, load_raw_csv  # noqa: E402

CLUSTER_RADIUS_M = 10_000.0
DEFAULT_OUT_ROOT = _PROJECT / "outputs" / "network_graph_10km_2018_2026"
YEARS_DEFAULT = list(range(2018, 2027))  # 2018–2026


@dataclass(frozen=True)
class MergeUnit:
    pkl_stem: str
    display_name: str
    member_abbr: tuple[str, ...]


MERGE_UNITS: tuple[MergeUnit, ...] = (
    MergeUnit("Maryland", "Maryland", ("MD", "DE", "DC")),
    MergeUnit("Dakotas", "Dakotas", ("SD", "ND")),
    MergeUnit("Connecticut", "Connecticut", ("CT", "RI")),
)

MERGED_ALL_MEMBERS = frozenset(a for u in MERGE_UNITS for a in u.member_abbr)

MIN_NODES_DEFAULT = 3
# Extreme small states at 10 km: allow 1+ hypernodes so merges always succeed.
MIN_NODES_BY_ABBR = {"DC": 1, "DE": 1, "ND": 1, "SD": 1, "RI": 1}


def _graph_from_clusters(
    centers: np.ndarray,
    cluster_caps: np.ndarray,
    cluster_counts: np.ndarray,
    cluster_l1: np.ndarray,
    cluster_l2: np.ndarray,
    cluster_dc: np.ndarray,
    coords_3857: np.ndarray,
    state: str,
    n_raw: int,
    radius_m: float,
) -> tuple[nx.Graph, dict]:
    n_nodes = len(centers)
    if n_nodes == 0:
        return nx.Graph(), {
            "state": state,
            "n_raw": n_raw,
            "n_nodes": 0,
            "n_edges": 0,
            "cluster_radius_m": float(radius_m),
            "dist_max_m": 1.0,
            "adjacency": "empty",
        }

    dist_matrix = np.linalg.norm(coords_3857[:, None] - coords_3857[None, :], axis=2)
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
            float(cluster_l1[i]),
            float(cluster_l2[i]),
            float(cluster_dc[i]),
        )
    for a in range(n_nodes):
        for b in range(a + 1, n_nodes):
            d_m = float(dist_matrix[a, b])
            G.add_edge(a, b, weight=d_m / dist_max, distance_m=d_m)

    return G, {
        "state": state,
        "n_raw": n_raw,
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "cluster_radius_m": float(radius_m),
        "dist_max_m": dist_max,
        "adjacency": "complete_small" if n_nodes < 4 else "voronoi",
    }


def build_state_graph_flex(
    df: pd.DataFrame,
    state: str,
    radius_m: float,
) -> tuple[nx.Graph | None, dict | None]:
    mn = MIN_NODES_BY_ABBR.get(state, MIN_NODES_DEFAULT)
    sub = df[df["state"] == state]
    if len(sub) < 1:
        return None, None

    coords_ll = sub[["lat", "lon"]].values.astype(np.float64)
    capacities = sub["capacity"].values.astype(np.float64)
    n_l1 = sub["n_l1"].values.astype(np.float64)
    n_l2 = sub["n_l2"].values.astype(np.float64)
    n_dc = sub["n_dc"].values.astype(np.float64)
    n_raw = len(sub)
    centers, cluster_caps, cluster_counts, cl1, cl2, cdc = s2.cluster_stations(
        coords_ll, capacities, radius_m=radius_m, n_l1=n_l1, n_l2=n_l2, n_dc=n_dc
    )
    n_nodes = len(centers)
    if n_nodes < mn:
        return None, None

    coords_3857 = s2._to_epsg3857_meters(centers[:, 0], centers[:, 1])

    if n_nodes >= 4:
        orig = s2.MIN_NODES
        s2.MIN_NODES = mn
        try:
            return s2.build_state_graph(df, state, radius_m=radius_m)
        finally:
            s2.MIN_NODES = orig

    G, stats = _graph_from_clusters(
        centers, cluster_caps, cluster_counts, cl1, cl2, cdc, coords_3857, state, n_raw, radius_m
    )
    return G, stats


def _empty_state_payload(abbr: str, year: int) -> dict:
    return {
        "network": nx.Graph(),
        "meta": {
            "state": abbr,
            "n_raw": 0,
            "n_nodes": 0,
            "n_edges": 0,
            "year": year,
            "cluster_radius_m": CLUSTER_RADIUS_M,
            "edge_weight": "normalized_distance",
            "adjacency": "empty",
        },
    }


def merge_payloads(by_abbr: dict[str, dict], unit: MergeUnit, year: int) -> dict:
    G_out = nx.Graph()
    offset = 0
    member_stats: list[dict] = []

    for abbr in unit.member_abbr:
        payload = by_abbr.get(abbr) or _empty_state_payload(abbr, year)
        G = payload["network"]
        mapping = {n: n + offset for n in G.nodes()}
        G_part = nx.relabel_nodes(G, mapping, copy=True)
        for node in G_part.nodes:
            G_part.nodes[node]["member_state"] = abbr
        G_out = nx.compose(G_out, G_part)
        member_stats.append(
            {
                "member": abbr,
                "n_nodes": int(G.number_of_nodes()),
                "n_edges": int(G.number_of_edges()),
            }
        )
        offset += G.number_of_nodes()

    return {
        "network": G_out,
        "meta": {
            "year": year,
            "unit": unit.pkl_stem,
            "display_name": unit.display_name,
            "member_states": list(unit.member_abbr),
            "n_nodes": int(G_out.number_of_nodes()),
            "n_edges": int(G_out.number_of_edges()),
            "cluster_radius_m": CLUSTER_RADIUS_M,
            "edge_weight": "normalized_distance",
            "adjacency": "voronoi",
            "merge_method": "disjoint_union",
            "node_features": "lat,lon,n_l1,n_l2,n_dc,capacity,n_stations",
            "capacity_formula": "5*n_l1+25*n_l2+300*n_dc",
            "members": member_stats,
        },
    }


def ensure_stations_csv(year: int, processed_dir: Path) -> Path:
    out = processed_dir / f"stations_{year}_48states.csv"
    if out.is_file():
        return out
    raw = default_csv_path(_PROJECT, year)
    if not raw.is_file():
        raise FileNotFoundError(f"Missing raw AFDC CSV for {year}: {raw}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{year}] Export Step1 -> {out.name}")
    prepare_stations(load_raw_csv(raw)).to_csv(out, index=False)
    return out


def build_year(year: int, out_root: Path, processed_dir: Path, verbose: bool = True) -> pd.DataFrame:
    stations_path = ensure_stations_csv(year, processed_dir)
    df = load_step1_stations(stations_path)
    year_dir = out_root / str(year)
    net_dir = year_dir / "network_structures"
    net_dir.mkdir(parents=True, exist_ok=True)

    by_abbr: dict[str, dict] = {}
    rows: list[dict] = []

    if verbose:
        print(f"\n=== {year} | cluster={CLUSTER_RADIUS_M / 1000:.0f} km ===")
        print(f"Stations: {stations_path} ({len(df):,} rows)")

    for abbr in STATES_ORDER:
        G, stats = build_state_graph_flex(df, abbr, radius_m=CLUSTER_RADIUS_M)
        if G is None or stats is None:
            if verbose:
                print(f"  {abbr}: skip (no graph)")
            continue
        by_abbr[abbr] = {
            "network": G,
            "meta": {
                **stats,
                "year": year,
                "edge_weight": "normalized_distance",
                "node_features": "lat,lon,n_l1,n_l2,n_dc,capacity,n_stations",
                "capacity_formula": "5*n_l1+25*n_l2+300*n_dc",
            },
        }

    for abbr in STATES_ORDER:
        if abbr in MERGED_ALL_MEMBERS or abbr not in by_abbr:
            continue
        pkl_path = net_dir / f"network_{abbr}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(by_abbr[abbr], f)
        st = by_abbr[abbr]["meta"]
        rows.append(
            {
                "year": year,
                "unit": abbr,
                "display_name": abbr,
                "members": abbr,
                "n_raw": st["n_raw"],
                "n_nodes": st["n_nodes"],
                "n_edges": st["n_edges"],
                "cluster_radius_m": CLUSTER_RADIUS_M,
            }
        )
        if verbose:
            print(f"  {abbr}: nodes={st['n_nodes']} edges={st['n_edges']}")

    for unit in MERGE_UNITS:
        payload = merge_payloads(by_abbr, unit, year)
        with open(net_dir / f"network_{unit.pkl_stem}.pkl", "wb") as f:
            pickle.dump(payload, f)
        meta = payload["meta"]
        rows.append(
            {
                "year": year,
                "unit": unit.pkl_stem,
                "display_name": unit.display_name,
                "members": ",".join(unit.member_abbr),
                "n_raw": sum(
                    by_abbr.get(m, {"meta": {"n_raw": 0}})["meta"]["n_raw"]
                    for m in unit.member_abbr
                ),
                "n_nodes": meta["n_nodes"],
                "n_edges": meta["n_edges"],
                "cluster_radius_m": CLUSTER_RADIUS_M,
            }
        )
        if verbose:
            parts = ", ".join(
                f"{m}={by_abbr[m]['meta']['n_nodes']}"
                for m in unit.member_abbr
                if m in by_abbr
            )
            print(f"  {unit.display_name} ({parts}) -> nodes={meta['n_nodes']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(year_dir / "network_summary.csv", index=False)
    if verbose:
        print(f"  units written: {len(summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="10 km networks 2018–2026 with fixed merges")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = _PROJECT / out_root
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = _PROJECT / processed_dir

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.txt").write_text(
        "10 km hypernode clustering; edge weight w_ij = d_ij / d_max.\n"
        "Node features: n_l1,n_l2,n_dc,capacity (capacity=5*L1+25*L2+300*DC summed in cluster).\n"
        "Merges (all years):\n"
        "  Maryland    = DC + DE + MD\n"
        "  Dakotas     = SD + ND\n"
        "  Connecticut = RI + CT\n"
        "Layout: <year>/network_structures/network_<unit>.pkl\n",
        encoding="utf-8",
    )

    parts = [build_year(y, out_root, processed_dir) for y in sorted(args.years)]
    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(out_root / "panel_hypernode_counts.csv", index=False)

    print(f"\n[done] {out_root.resolve()}")
    print(f"       panel_hypernode_counts.csv ({len(panel)} rows)")


if __name__ == "__main__":
    main()
