#!/usr/bin/env python3
"""
Build the 2018–2023 analytical-unit network panel with 10 km hypernodes.

Per unit
--------
- Deterministic greedy 10 km clustering using WGS84 geodesic distance
- Voronoi adjacency in a local azimuthal-equidistant projection
- Edge distance d_ij is WGS84 geodesic distance

Combined analytical units (all years)
------------------------
- Maryland    : DC + DE + MD -> network_Maryland.pkl
- Dakotas     : SD + ND      -> network_Dakotas.pkl
- Connecticut : RI + CT      -> network_Connecticut.pkl

Member-state station rows are combined before clustering and Voronoi
construction; combined units are not disjoint unions of state graphs.

Output (default)
----------------
  outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023/
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
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from build_charging_network_step2 import (  # noqa: E402
    build_region_graph,
    load_step1_stations,
)
from prepare_stations import default_csv_path, prepare_stations  # noqa: E402
from run_charging_network import STATES_ORDER, load_raw_csv  # noqa: E402

CLUSTER_RADIUS_M = 10_000.0
NETWORK_BUILD_DATE = "2026-10-29"
NETWORK_VERSION = "1029-geodesic-v1"
DEFAULT_OUT_ROOT = (
    _PROJECT
    / "outputs"
    / "1029-network-data-preprocessing-10km-geodesic-2018-2023"
)
YEARS_DEFAULT = list(range(2018, 2024))


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


def build_state_graph_flex(
    df: pd.DataFrame,
    state: str,
    radius_m: float,
) -> tuple[nx.Graph | None, dict | None]:
    mn = MIN_NODES_BY_ABBR.get(state, MIN_NODES_DEFAULT)
    return build_region_graph(
        df,
        [state],
        state,
        radius_m=radius_m,
        min_nodes=mn,
    )


def ensure_stations_csv(year: int, processed_dir: Path) -> Path:
    out = processed_dir / f"stations_{year}_48states.csv"
    if out.is_file():
        return out
    raw = default_csv_path(_PROJECT, year)
    if not raw.is_file():
        raise FileNotFoundError(
            f"Need Step-1 stations for {year}. Missing both:\n"
            f"  processed: {out}\n"
            f"  raw AFDC:  {raw}\n"
            f"For CF scenario K on the cluster, upload/copy "
            f"data/processed/stations_{year}_48states.csv "
            f"(same file used to build the 10 km 2023 nets). "
            f"Raw historical AFDC is not required if the processed CSV exists."
        )
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
        if abbr in MERGED_ALL_MEMBERS:
            continue
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
                "network_version": NETWORK_VERSION,
                "network_build_date": NETWORK_BUILD_DATE,
                "edge_weight": "normalized_geodesic_distance",
                "analysis_adjacency": "unweighted",
                "node_features": "lat,lon,n_l1,n_l2,n_dc,capacity,n_stations",
                "capacity_formula": "5*n_l1+25*n_l2+300*n_dc",
            },
        }

    for abbr in STATES_ORDER:
        if abbr not in by_abbr:
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
                "network_version": NETWORK_VERSION,
                "network_build_date": NETWORK_BUILD_DATE,
                "cluster_distance": st["cluster_distance"],
                "adjacency": st["adjacency"],
                "is_connected": nx.is_connected(by_abbr[abbr]["network"]),
                "n_components": nx.number_connected_components(by_abbr[abbr]["network"]),
            }
        )
        if verbose:
            print(f"  {abbr}: nodes={st['n_nodes']} edges={st['n_edges']}")

    for unit in MERGE_UNITS:
        G, stats = build_region_graph(
            df,
            list(unit.member_abbr),
            unit.pkl_stem,
            radius_m=CLUSTER_RADIUS_M,
            min_nodes=1,
        )
        if G is None or stats is None:
            raise RuntimeError(f"Could not build combined unit {unit.pkl_stem} for {year}")
        payload = {
            "network": G,
            "meta": {
                **stats,
                "year": year,
                "network_version": NETWORK_VERSION,
                "network_build_date": NETWORK_BUILD_DATE,
                "unit": unit.pkl_stem,
                "display_name": unit.display_name,
                "member_states": list(unit.member_abbr),
                "edge_weight": "normalized_geodesic_distance",
                "analysis_adjacency": "unweighted",
                "merge_method": "combine_stations_before_clustering",
                "node_features": "lat,lon,n_l1,n_l2,n_dc,capacity,n_stations",
                "capacity_formula": "5*n_l1+25*n_l2+300*n_dc",
            },
        }
        with open(net_dir / f"network_{unit.pkl_stem}.pkl", "wb") as f:
            pickle.dump(payload, f)
        meta = payload["meta"]
        rows.append(
            {
                "year": year,
                "unit": unit.pkl_stem,
                "display_name": unit.display_name,
                "members": ",".join(unit.member_abbr),
                "n_raw": meta["n_raw"],
                "n_nodes": meta["n_nodes"],
                "n_edges": meta["n_edges"],
                "cluster_radius_m": CLUSTER_RADIUS_M,
                "network_version": NETWORK_VERSION,
                "network_build_date": NETWORK_BUILD_DATE,
                "cluster_distance": meta["cluster_distance"],
                "adjacency": meta["adjacency"],
                "is_connected": nx.is_connected(G),
                "n_components": nx.number_connected_components(G),
            }
        )
        if verbose:
            members = "+".join(unit.member_abbr)
            print(f"  {unit.display_name} ({members}) -> nodes={meta['n_nodes']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(year_dir / "network_summary.csv", index=False)
    if verbose:
        print(f"  units written: {len(summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2018-2023 10 km geodesic network-data preprocessing"
    )
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
        f"Network version: {NETWORK_VERSION}\n"
        f"Method date: {NETWORK_BUILD_DATE}\n"
        "Deterministic 10 km hypernode clustering using WGS84 geodesic distance.\n"
        "Voronoi adjacency uses a local azimuthal-equidistant projection per unit.\n"
        "Canonical efficiency analysis treats edges as unweighted. Stored edge distances are WGS84 geodesic.\n"
        "Node features: n_l1,n_l2,n_dc,capacity (capacity=5*L1+25*L2+300*DC summed in cluster).\n"
        "Combined units (stations combined before clustering and graph construction):\n"
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
