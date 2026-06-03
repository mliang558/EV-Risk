#!/usr/bin/env python3
"""
Rebuild windows_nodes.parquet after fixing BallTree index/dist swap bug.

Keeps windows_meta.csv unchanged; recomputes node membership per window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from census_tract_strata import assign_nodes_to_tract_strata, load_or_build_tract_strata, nodes_strata_cache_path
from sample_sliding_windows_step3 import (
    EARTH_RADIUS_KM,
    N_MAX,
    N_MIN,
    STRATUM_CONFIG,
    load_national_nodes,
)


def rebuild_nodes(
    windows_dir: Path,
    network_dir: Path,
    project_root: Path,
    geocode_method: str = "county",
) -> pd.DataFrame:
    meta = pd.read_csv(windows_dir / "windows_meta.csv")
    nodes = load_national_nodes(network_dir)
    tract_strata = load_or_build_tract_strata(project_root)
    nodes_cache = nodes_strata_cache_path(project_root, geocode_method)
    if nodes_cache.exists():
        nodes = pd.read_parquet(nodes_cache)
    else:
        nodes = assign_nodes_to_tract_strata(nodes, tract_strata, project_root, geocode_method=geocode_method)

    coords_rad = np.radians(nodes[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    rows = []

    for _, wrow in meta.iterrows():
        wid = wrow["window_id"]
        stratum = wrow["window_stratum"]
        radius_km = float(wrow["radius_km"])
        radius_rad = radius_km / EARTH_RADIUS_KM
        seed_id = wrow["seed_node_id"]
        seed = nodes[nodes["node_id"] == seed_id].iloc[0]
        lat, lon = float(seed["lat"]), float(seed["lon"])

        inds, _ = tree.query_radius(
            np.radians([[lat, lon]]),
            r=radius_rad,
            return_distance=True,
            sort_results=True,
        )
        pick = inds[0]
        if len(pick) < N_MIN:
            continue
        if len(pick) > N_MAX:
            rng = np.random.default_rng(abs(hash(wid)) % (2**31))
            pick = rng.choice(pick, size=N_MAX, replace=False)

        for idx in pick:
            nrow = nodes.iloc[int(idx)]
            rows.append(
                {
                    "window_id": wid,
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

    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    windows_dir = project_root / "outputs/window_samples_2026_step3"
    network_dir = project_root / "outputs/network_graph_2026_step2/network_structures"

    print("Rebuilding windows_nodes.parquet ...")
    out = rebuild_nodes(windows_dir, network_dir, project_root)
    path = windows_dir / "windows_nodes.parquet"
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows, {out.window_id.nunique()} windows)")
    chk = out.groupby("window_id").node_id.nunique()
    print(f"Nodes per window: min={chk.min()} median={chk.median():.0f} max={chk.max()}")


if __name__ == "__main__":
    main()
