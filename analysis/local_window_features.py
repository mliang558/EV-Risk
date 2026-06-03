"""Per-window features aggregated from node-level subgraph attributes (no graph topology)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

LOCAL_FEATURE_NAMES = (
    "n_nodes",
    "total_capacity",
    "mean_capacity",
    "std_capacity",
    "max_capacity",
    "mean_lat",
    "mean_lon",
    "std_lat",
    "std_lon",
    "mean_dcfc",
    "max_dcfc",
    "mean_degree",
    "max_degree",
)


def _lat_lon_dcfc_from_npz(data, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step-4 npz often has only capacity+edges; lat/lon may be in x or absent."""
    if "lat" in data.files and "lon" in data.files:
        return (
            np.asarray(data["lat"], dtype=np.float64),
            np.asarray(data["lon"], dtype=np.float64),
            np.asarray(data["dcfc_ratio"], dtype=np.float64) if "dcfc_ratio" in data.files else np.zeros(n),
        )
    if "x" in data.files:
        x = np.asarray(data["x"], dtype=np.float64)
        if x.ndim == 2 and x.shape[0] == n and x.shape[1] >= 4:
            return x[:, 1], x[:, 0], x[:, 3]  # lon, lat, dcfc in node_features order
        if x.ndim == 2 and x.shape[0] == n and x.shape[1] >= 2:
            return x[:, 0], x[:, 1], np.zeros(n)
    return np.zeros(n), np.zeros(n), np.zeros(n)


def local_features_from_npz(data, meta: dict[str, Any] | None = None) -> dict[str, float]:
    """Fast local stats from Step-4 subgraph npz (no NetworkX)."""
    n = int(data["n_nodes"])
    caps = np.asarray(data["capacity"], dtype=np.float64)
    lats, lons, dcfc = _lat_lon_dcfc_from_npz(data, n)

    # Fallback: window center from windows_meta when per-node lat/lon not in npz
    if meta is not None and (lats.sum() == 0 and lons.sum() == 0):
        clat = meta.get("center_lat")
        clon = meta.get("center_lon")
        if clat is not None and clon is not None:
            lats = np.full(n, float(clat), dtype=np.float64)
            lons = np.full(n, float(clon), dtype=np.float64)

    feats: dict[str, float] = {
        "n_nodes": float(n),
        "total_capacity": float(caps.sum()) if n else 0.0,
        "mean_capacity": float(caps.mean()) if n else 0.0,
        "std_capacity": float(caps.std()) if n > 1 else 0.0,
        "max_capacity": float(caps.max()) if n else 0.0,
        "mean_lat": float(lats.mean()) if n else 0.0,
        "mean_lon": float(lons.mean()) if n else 0.0,
        "std_lat": float(lats.std()) if n > 1 else 0.0,
        "std_lon": float(lons.std()) if n > 1 else 0.0,
        "mean_dcfc": float(dcfc.mean()) if n else 0.0,
        "max_dcfc": float(dcfc.max()) if n else 0.0,
        "mean_degree": 0.0,
        "max_degree": 0.0,
    }
    if n > 0 and "edge_index" in data.files:
        ei = data["edge_index"]
        deg = np.zeros(n, dtype=np.float64)
        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            if u == v:
                continue
            deg[u] += 1
            deg[v] += 1
        feats["mean_degree"] = float(deg.mean())
        feats["max_degree"] = float(deg.max())
    if meta is not None:
        if "radius_km" in meta and "radius_km" not in feats:
            feats["radius_km"] = float(meta.get("radius_km", 0))
    return feats


def build_local_feature_table(
    window_ids: list[str],
    data_dir: Path,
    *,
    cache_csv: Path | None = None,
    windows_meta: "pd.DataFrame | None" = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    import pandas as pd

    from gnn.global_graph_features import subgraph_path_for_window

    if cache_csv is not None and cache_csv.is_file():
        if verbose:
            print(f"[local_feat] load {cache_csv}", flush=True)
        return pd.read_csv(cache_csv)

    meta_by_wid: dict[str, dict] = {}
    if windows_meta is not None and "window_id" in windows_meta.columns:
        for _, row in windows_meta.iterrows():
            meta_by_wid[str(row["window_id"])] = row.to_dict()

    data_dir = data_dir.resolve()
    rows = []
    it = window_ids
    if verbose and tqdm is not None:
        it = tqdm(window_ids, desc="Local node feats", unit="win", file=__import__("sys").stderr)
    for wid in it:
        sp = subgraph_path_for_window(data_dir, str(wid))
        if sp is None:
            continue
        z = np.load(sp, allow_pickle=True)
        f = local_features_from_npz(z, meta=meta_by_wid.get(str(wid)))
        f["window_id"] = str(wid)
        rows.append(f)
    feat_df = pd.DataFrame(rows)
    if cache_csv is not None:
        feat_df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"[local_feat] cached -> {cache_csv}", flush=True)
    return feat_df
