"""Node features (lon, lat, capacity, DCFC ratio) aligned with Step-4 subgraphs."""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

def _ensure_analysis_path() -> None:
    here = Path(__file__).resolve()
    for base in (here.parents[1], here.parents[2], Path.cwd()):
        if (base / "build_charging_network_step2.py").exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            return
    raise ImportError(
        "Need build_charging_network_step2.py on PYTHONPATH (copy from analysis/ next to gnn/)"
    )


_ensure_analysis_path()
from build_charging_network_step2 import cluster_stations, load_step1_stations
from run_charging_network import STATES_ORDER, _to_epsg3857_meters

CLUSTER_RADIUS_M = 200.0


def parse_node_id(node_id: str) -> tuple[str, int]:
    m = re.match(r"^([A-Z]{2})_(\d+)$", str(node_id))
    if not m:
        raise ValueError(f"Bad node_id: {node_id}")
    return m.group(1), int(m.group(2))


def load_state_graph(network_dir: Path, state: str, cache: dict) -> nx.Graph:
    if state in cache:
        return cache[state]
    pkl = network_dir / f"network_{state}.pkl"
    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    G = payload["network"] if isinstance(payload, dict) else payload
    cache[state] = G
    return G


def build_dcfc_lookup(stations_csv: Path, radius_m: float = CLUSTER_RADIUS_M) -> pd.DataFrame:
    """
    Per hypernode (state, local_id): DCFC ratio = n_dc / (n_l1 + n_l2 + n_dc).
    Uses the same 200 m greedy clustering as Step 2.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    print("[Step5] DCFC: loading stations + clustering 49 states...", flush=True)
    df = load_step1_stations(stations_csv)
    rows = []
    for state in tqdm(STATES_ORDER, desc="DCFC lookup by state", file=sys.stderr, mininterval=0.3):
        sub = df[df["state"] == state]
        if len(sub) < 3:
            continue
        coords = sub[["lat", "lon"]].values.astype(np.float64)
        caps = sub["capacity"].values.astype(np.float64)
        centers, _, _ = cluster_stations(coords, caps, radius_m=radius_m)

        coords_m = _to_epsg3857_meters(coords[:, 0], coords[:, 1])
        centers_m = _to_epsg3857_meters(centers[:, 0], centers[:, 1])
        n_dc = sub["n_dc"].values.astype(np.float64)
        n_l1 = sub["n_l1"].values.astype(np.float64)
        n_l2 = sub["n_l2"].values.astype(np.float64)

        for cid in range(len(centers)):
            d = np.linalg.norm(coords_m - centers_m[cid], axis=1)
            mask = d <= radius_m
            denom = float(n_l1[mask].sum() + n_l2[mask].sum() + n_dc[mask].sum())
            ratio = float(n_dc[mask].sum() / denom) if denom > 0 else 0.0
            rows.append(
                {
                    "state": state,
                    "local_id": cid,
                    "dcfc_ratio": np.clip(ratio, 0.0, 1.0),
                }
            )
    out = pd.DataFrame(rows)
    print(f"[Step5] DCFC lookup done ({len(out)} hypernodes).", flush=True)
    return out


def window_graph_from_nodes(
    wnodes: pd.DataFrame,
    network_dir: Path,
    cache: dict,
) -> tuple[nx.Graph, list[tuple[str, int]]]:
    """Rebuild window graph with same relabeling order as step4_export."""
    by_state: dict[str, list[int]] = {}
    for _, row in wnodes.iterrows():
        st = str(row["state"])
        lid = int(row["local_id"])
        by_state.setdefault(st, []).append(lid)

    components: list[nx.Graph] = []
    for st, lids in by_state.items():
        if st not in STATES_ORDER:
            continue
        Gs = load_state_graph(network_dir, st, cache)
        sub = Gs.subgraph(lids).copy()
        if sub.number_of_nodes() < 1:
            continue
        for n in sub.nodes():
            sub.nodes[n]["state"] = st
        components.append(sub)

    if not components:
        raise ValueError("Empty window graph")
    G = components[0] if len(components) == 1 else nx.disjoint_union_all(components)

    old_nodes = list(G.nodes())
    old_keys = []
    for old in old_nodes:
        st = G.nodes[old].get("state", "")
        old_keys.append((st, int(old)))

    mapping = {old: i for i, old in enumerate(old_nodes)}
    H = nx.relabel_nodes(G, mapping)
    return H, old_keys


def node_feature_matrix(
    G: nx.Graph,
    old_keys: list[tuple[str, int]],
    dcfc_lookup: pd.DataFrame | None,
) -> np.ndarray:
    """Shape (n, 4): lon, lat, capacity, dcfc_ratio."""
    dcfc_map = {}
    if dcfc_lookup is not None and not dcfc_lookup.empty:
        for _, r in dcfc_lookup.iterrows():
            dcfc_map[(r["state"], int(r["local_id"]))] = float(r["dcfc_ratio"])

    feats = []
    for i in range(G.number_of_nodes()):
        st, lid = old_keys[i]
        d = G.nodes[i]
        lat = float(d.get("lat", 0.0))
        lon = float(d.get("lon", 0.0))
        cap = float(d.get("capacity", 0.0))
        dcfc = dcfc_map.get((st, lid), 0.0)
        feats.append([lon, lat, cap, dcfc])
    return np.asarray(feats, dtype=np.float32)


def features_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if all(k in data.files for k in ("lat", "lon", "capacity", "dcfc_ratio")):
        return np.stack(
            [
                data["lon"].astype(np.float32),
                data["lat"].astype(np.float32),
                data["capacity"].astype(np.float32),
                data["dcfc_ratio"].astype(np.float32),
            ],
            axis=1,
        )
    return None
