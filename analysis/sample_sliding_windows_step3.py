#!/usr/bin/env python3
"""
Step 3: Stratified sliding-window sampling (Census tract urban / suburban / rural).

Strata
------
Join each charging node to a 2020 Census tract (pygris) and label the tract using:
  1) Cached tract table (Census 2020 PL urban-pop share via API), or
  2) CBSA fallback (metro / micropolitan) if no API key.

Sampling (per stratum)
----------------------
  - Random seed node from that stratum
  - Radius: Urban 10 km | Suburban 30 km | Rural 80 km
  - All national nodes within R (not only same-stratum nodes)
  - N_min <= n <= N_max (else subsample to N_max or discard)
  - Independent centers: distance > R/2 from prior centers in the same stratum

Outputs
-------
  outputs/window_samples_2026_step3/windows_meta.csv
  outputs/window_samples_2026_step3/windows_nodes.parquet
  outputs/window_samples_2026_step3/sampling_report.json

Usage:
  python analysis/build_census_tract_strata.py   # optional: needs CENSUS_API_KEY
  python analysis/sample_sliding_windows_step3.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from census_tract_strata import assign_nodes_to_tract_strata, load_or_build_tract_strata
from run_charging_network import STATES_ORDER

try:
    from sklearn.neighbors import BallTree
except ImportError as e:
    raise ImportError("Step 3 requires scikit-learn.") from e


EARTH_RADIUS_KM = 6371.0

STRATUM_CONFIG = {
    "urban": {"radius_km": 10.0, "target": 3334},
    "suburban": {"radius_km": 30.0, "target": 3333},
    "rural": {"radius_km": 80.0, "target": 3333},
}

N_MIN = 15  # raised from 5: small windows → unstable betweenness / noisy Y
N_MAX = 100

TARGET_TOTAL_WINDOWS = 10000
STRATUM_TARGETS = {
    "urban": 3334,
    "suburban": 3333,
    "rural": 3333,
}


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_national_nodes(network_dir: Path) -> pd.DataFrame:
    rows = []
    for state in STATES_ORDER:
        pkl_path = network_dir / f"network_{state}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        G = payload["network"] if isinstance(payload, dict) else payload
        for local_id, data in G.nodes(data=True):
            rows.append(
                {
                    "node_id": f"{state}_{local_id}",
                    "state": state,
                    "local_id": int(local_id),
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "capacity": float(data.get("capacity", 0.0)),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No network pickles found in {network_dir}")
    return pd.DataFrame(rows)


def _centers_too_close(
    lat: float,
    lon: float,
    accepted: list[tuple[float, float]],
    min_sep_km: float,
) -> bool:
    if not accepted:
        return False
    lats = np.array([c[0] for c in accepted])
    lons = np.array([c[1] for c in accepted])
    return bool(np.any(haversine_km(lat, lon, lats, lons) <= min_sep_km))


def sample_stratum(
    nodes: pd.DataFrame,
    stratum: str,
    radius_km: float,
    target: int,
    rng: np.random.Generator,
    n_min: int,
    n_max: int,
    max_attempts_factor: int = 120,
    min_sep_fractions: tuple[float, ...] = (0.5, 0.35),
    *,
    initial_centers: list[tuple[float, float]] | None = None,
    window_id_start: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    seed_pool = nodes[nodes["stratum"] == stratum]
    if seed_pool.empty:
        return [], [], {"stratum": stratum, "error": "empty_seed_pool"}

    all_coords_rad = np.radians(nodes[["lat", "lon"]].values)
    tree = BallTree(all_coords_rad, metric="haversine")
    radius_rad = radius_km / EARTH_RADIUS_KM
    seed_indices = seed_pool.index.to_numpy()
    accepted_centers: list[tuple[float, float]] = []
    meta_rows: list[dict] = []
    node_rows: list[dict] = []
    attempts = 0
    max_attempts = max(target * max_attempts_factor, target * 20)
    passes_log: list[dict] = []

    for pass_i, sep_frac in enumerate(min_sep_fractions):
        if len(meta_rows) >= target:
            break
        min_sep_km = radius_km * sep_frac
        order = rng.permutation(len(seed_indices))
        cursor = 0
        pass_start = len(meta_rows)
        pass_attempts = 0
        pass_cap = max_attempts if pass_i == 0 else max_attempts * 2

        while len(meta_rows) < target and pass_attempts < pass_cap:
            if cursor >= len(order):
                cursor = 0
                rng.shuffle(order)
            seed_idx = seed_indices[order[cursor]]
            cursor += 1
            pass_attempts += 1
            attempts += 1

            seed = nodes.loc[seed_idx]
            lat, lon = float(seed["lat"]), float(seed["lon"])
            if _centers_too_close(lat, lon, accepted_centers, min_sep_km):
                continue

            inds, _dist = tree.query_radius(
                np.radians([[lat, lon]]),
                r=radius_rad,
                return_distance=True,
                sort_results=True,
            )
            global_inds = inds[0]
            if len(global_inds) < n_min:
                continue

            n_raw = len(global_inds)
            pick = (
                rng.choice(global_inds, size=n_max, replace=False)
                if n_raw > n_max
                else global_inds
            )

            window_nodes = nodes.iloc[pick]
            window_id = f"w_{stratum}_{window_id_start + len(meta_rows):05d}"
            accepted_centers.append((lat, lon))

            stratum_mix = window_nodes["stratum"].value_counts(normalize=True).to_dict()
            meta_rows.append(
                {
                    "window_id": window_id,
                    "window_stratum": stratum,
                    "radius_km": radius_km,
                    "min_center_sep_km": min_sep_km,
                    "overlap_pass": pass_i + 1,
                    "seed_node_id": seed["node_id"],
                    "seed_tract_geoid": seed.get("tract_geoid", ""),
                    "stratum_source": seed.get("source", ""),
                    "center_lat": lat,
                    "center_lon": lon,
                    "n_nodes_raw": n_raw,
                    "n_nodes": len(pick),
                    "frac_urban": stratum_mix.get("urban", 0.0),
                    "frac_suburban": stratum_mix.get("suburban", 0.0),
                    "frac_rural": stratum_mix.get("rural", 0.0),
                }
            )
            for _, nrow in window_nodes.iterrows():
                node_rows.append(
                    {
                        "window_id": window_id,
                        "node_id": nrow["node_id"],
                        "state": nrow["state"],
                        "local_id": nrow["local_id"],
                        "lat": nrow["lat"],
                        "lon": nrow["lon"],
                        "capacity": nrow["capacity"],
                        "node_stratum": nrow["stratum"],
                        "tract_geoid": nrow.get("tract_geoid", ""),
                    }
                )

        passes_log.append(
            {
                "pass": pass_i + 1,
                "min_sep_fraction": sep_frac,
                "accepted": len(meta_rows) - pass_start,
                "attempts": pass_attempts,
            }
        )

    stats = {
        "stratum": stratum,
        "radius_km": radius_km,
        "target": target,
        "accepted": len(meta_rows),
        "attempts": attempts,
        "seed_pool_nodes": len(seed_pool),
        "success_rate": len(meta_rows) / max(attempts, 1),
        "passes": passes_log,
    }
    return meta_rows, node_rows, stats


def run_sampling(
    network_dir: Path,
    out_dir: Path,
    stratum_config: dict,
    seed: int,
    n_min: int,
    n_max: int,
    max_attempts_factor: int,
    geocode_method: str = "fcc",
    verbose: bool = True,
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading networks from: {network_dir}")
    nodes = load_national_nodes(network_dir)
    if verbose:
        print(f"  National nodes: {len(nodes):,}")

    tract_strata = load_or_build_tract_strata(project_root)
    if verbose:
        print(f"  Tract strata source: {tract_strata['source'].iloc[0]}")
    nodes = assign_nodes_to_tract_strata(
        nodes, tract_strata, project_root, geocode_method=geocode_method
    )
    if verbose:
        print("  Nodes by tract stratum:")
        print(nodes["stratum"].value_counts().to_string())

    rng = np.random.default_rng(seed)
    all_meta: list[dict] = []
    all_nodes: list[dict] = []
    report: dict = {
        "n_min": n_min,
        "n_max": n_max,
        "seed": seed,
        "strata_source": str(tract_strata["source"].iloc[0]),
        "geocode_method": geocode_method,
        "strata": {},
    }

    for stratum, cfg in stratum_config.items():
        if verbose:
            print(f"\nSampling {stratum}: R={cfg['radius_km']} km, target={cfg['target']}")
        meta, nrows, stats = sample_stratum(
            nodes,
            stratum=stratum,
            radius_km=float(cfg["radius_km"]),
            target=int(cfg["target"]),
            rng=rng,
            n_min=n_min,
            n_max=n_max,
            max_attempts_factor=max_attempts_factor,
        )
        all_meta.extend(meta)
        all_nodes.extend(nrows)
        report["strata"][stratum] = stats
        if verbose:
            print(f"  Accepted {stats.get('accepted', 0)}/{stats.get('target', 0)} ({stats.get('attempts', 0)} attempts)")

    meta_df = pd.DataFrame(all_meta)
    nodes_df = pd.DataFrame(all_nodes)
    meta_path = out_dir / "windows_meta.csv"
    nodes_path = out_dir / "windows_nodes.parquet"
    meta_df.to_csv(meta_path, index=False)
    nodes_df.to_parquet(nodes_path, index=False)

    report["total_windows"] = len(meta_df)
    report["total_window_node_records"] = len(nodes_df)
    report_path = out_dir / "sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\nWrote {meta_path} ({len(meta_df):,} windows)")
        print(f"Wrote {nodes_path} ({len(nodes_df):,} node records)")
        print(f"Wrote {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: census-tract stratified sliding windows")
    parser.add_argument(
        "--network-dir",
        type=str,
        default="outputs/network_graph_2026_step2/network_structures",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/window_samples_2026_step3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-min", type=int, default=N_MIN)
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--target-per-stratum", type=int, default=None)
    parser.add_argument("--max-attempts-factor", type=int, default=120)
    parser.add_argument("--rebuild-tract-cache", action="store_true")
    parser.add_argument(
        "--geocode",
        choices=("fcc", "county"),
        default="fcc",
        help="fcc=FCC Census geocoder (tract GEOID); county=fast county modal strata",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    network_dir = Path(args.network_dir)
    if not network_dir.is_absolute():
        network_dir = project_root / network_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    if not network_dir.exists():
        print(f"Network dir not found: {network_dir}", file=sys.stderr)
        sys.exit(1)

    if args.rebuild_tract_cache:
        load_or_build_tract_strata(project_root, force_rebuild=True)

    cfg = {k: dict(v) for k, v in STRATUM_CONFIG.items()}
    if args.target_per_stratum is not None:
        for k in cfg:
            cfg[k]["target"] = args.target_per_stratum

    run_sampling(
        network_dir=network_dir,
        out_dir=out_dir,
        stratum_config=cfg,
        seed=args.seed,
        n_min=args.n_min,
        n_max=args.n_max,
        max_attempts_factor=args.max_attempts_factor,
        geocode_method=args.geocode,
    )


if __name__ == "__main__":
    main()
