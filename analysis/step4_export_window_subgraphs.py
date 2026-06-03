#!/usr/bin/env python3
"""
Step 4 (data prep): export each Step-3 window as an induced subgraph for attack simulation.

Reads:
  - outputs/window_samples_2026_step3/windows_meta.csv
  - outputs/window_samples_2026_step3/windows_nodes.parquet
  - outputs/network_graph_2026_step2/network_structures/network_<STATE>.pkl

Writes:
  - data/step4_gpu/manifest.csv
  - data/step4_gpu/subgraphs/<window_id>.npz

Each .npz contains:
  - n_nodes, edge_index (2,E), edge_weight (E,), capacity (n,)
  - order_betweenness, order_capacity (local node ranks, high-first)
  - node_id_str (original ids), state, window_stratum, radius_km

Usage:
  python analysis/step4_export_window_subgraphs.py
  python analysis/step4_export_window_subgraphs.py --limit 100
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from run_charging_network import STATES_ORDER

_SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_path(arg: str | Path, *, cwd: Path | None = None) -> Path:
    """Relative paths resolve from cwd (where you run the command), not repo root."""
    p = Path(arg).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (cwd or Path.cwd()).resolve() / p


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


def induced_subgraph(
    G_full: nx.Graph,
    local_ids: list[int],
) -> nx.Graph | None:
    sub = G_full.subgraph(local_ids).copy()
    if sub.number_of_nodes() < 2:
        return None
    return sub


def relabel_graph(G: nx.Graph) -> tuple[nx.Graph, list[int], list[str]]:
    """Relabel nodes to 0..n-1; return mapping old_id list and node_id_str."""
    old_nodes = list(G.nodes())
    mapping = {old: i for i, old in enumerate(old_nodes)}
    H = nx.relabel_nodes(G, mapping)
    node_id_str = []
    for old in old_nodes:
        d = G.nodes[old]
        st = d.get("state", "")
        node_id_str.append(f"{st}_{old}" if st else str(old))
    return H, old_nodes, node_id_str


def attack_orders(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    b = nx.betweenness_centrality(G, weight="weight")
    order_b = np.array(sorted(G.nodes(), key=lambda u: -b.get(u, 0.0)), dtype=np.int32)
    cap = nx.get_node_attributes(G, "capacity")
    order_c = np.array(sorted(G.nodes(), key=lambda u: -float(cap.get(u, 0.0))), dtype=np.int32)
    return order_b, order_c


def graph_to_npz(
    G: nx.Graph,
    order_b: np.ndarray,
    order_c: np.ndarray,
    meta: dict,
) -> dict:
    n = G.number_of_nodes()
    caps = np.array([float(G.nodes[i].get("capacity", 0.0)) for i in range(n)], dtype=np.float32)
    edges = list(G.edges(data=True))
    if edges:
        src = np.array([u for u, v, _ in edges], dtype=np.int32)
        dst = np.array([v for u, v, _ in edges], dtype=np.int32)
        w = np.array([float(d.get("weight", 1.0)) for _, _, d in edges], dtype=np.float32)
        edge_index = np.stack([src, dst], axis=0)
        # undirected: add reverse
        edge_index = np.concatenate([edge_index, edge_index[::-1, [1, 0]]], axis=1)
        w = np.concatenate([w, w])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        w = np.zeros(0, dtype=np.float32)

    out = {
        "n_nodes": np.int32(n),
        "edge_index": edge_index,
        "edge_weight": w,
        "capacity": caps,
        "order_betweenness": order_b,
        "order_capacity": order_c,
    }
    for k, v in meta.items():
        if isinstance(v, (str, int, float)):
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Step-3 window subgraphs for Step-4")
    parser.add_argument(
        "--windows-dir",
        default="outputs/window_samples_2026_step3",
    )
    parser.add_argument(
        "--network-dir",
        default="outputs/network_graph_2026_step2/network_structures",
    )
    parser.add_argument("--out", default="data/step4_gpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cwd = Path.cwd()
    windows_dir = resolve_data_path(args.windows_dir, cwd=cwd)
    out_dir = resolve_data_path(args.out, cwd=cwd)
    try:
        from network_paths import resolve_network_dir

        network_dir = resolve_network_dir(args.network_dir, cwd=cwd)
    except ImportError:
        network_dir = resolve_data_path(args.network_dir, cwd=cwd)

    meta_path = windows_dir / "windows_meta.csv"
    nodes_path = windows_dir / "windows_nodes.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}\n"
            f"  cwd={cwd}\n"
            f"  Use: --windows-dir .  (when windows_meta.csv is in cwd)\n"
            f"  Or:  --windows-dir /opt/data_repo/mliang_work/step4_gpu"
        )

    sub_dir = out_dir / "subgraphs"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"windows_dir={windows_dir}", flush=True)
    print(f"network_dir={network_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    meta = pd.read_csv(meta_path)
    nodes = pd.read_parquet(nodes_path)
    if args.limit:
        meta = meta.head(args.limit)

    graph_cache: dict[str, nx.Graph] = {}
    manifest_rows = []
    skipped = 0

    for _, row in meta.iterrows():
        wid = row["window_id"]
        wnodes = nodes[nodes["window_id"] == wid]
        by_state: dict[str, list[int]] = {}
        for nid in wnodes["node_id"]:
            st, lid = parse_node_id(nid)
            by_state.setdefault(st, []).append(lid)

        # Build disconnected union per state then combine — windows are spatially local;
        # nodes should be mostly one state; if multi-state, merge graphs with disjoint union.
        components: list[nx.Graph] = []
        states_in = []
        for st, lids in by_state.items():
            if st not in STATES_ORDER:
                continue
            Gs = load_state_graph(network_dir, st, graph_cache)
            sub = induced_subgraph(Gs, lids)
            if sub is not None:
                H, _, _ = relabel_graph(sub)
                for n in H.nodes():
                    H.nodes[n]["state"] = st
                components.append(H)
                states_in.append(st)

        if not components:
            skipped += 1
            continue

        if len(components) == 1:
            G = components[0]
        else:
            G = nx.disjoint_union_all(components)

        if G.number_of_nodes() < 2:
            skipped += 1
            continue

        G, old_ids, _ = relabel_graph(G)
        order_b, order_c = attack_orders(G)
        npz_meta = {
            "window_id": wid,
            "window_stratum": str(row.get("window_stratum", "")),
            "radius_km": float(row.get("radius_km", 0)),
            "n_nodes_meta": int(row.get("n_nodes", G.number_of_nodes())),
        }
        arrays = graph_to_npz(G, order_b, order_c, npz_meta)
        np.savez_compressed(sub_dir / f"{wid}.npz", **arrays)

        manifest_rows.append(
            {
                "window_id": wid,
                "window_stratum": row.get("window_stratum"),
                "radius_km": row.get("radius_km"),
                "n_nodes": int(arrays["n_nodes"]),
                "n_edges": int(arrays["edge_index"].shape[1] // 2),
                "states": ",".join(sorted(set(states_in))),
                "subgraph_path": f"subgraphs/{wid}.npz",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    meta.to_csv(out_dir / "windows_meta.csv", index=False)

    print(f"Exported {len(manifest)} subgraphs -> {out_dir}")
    print(f"Skipped {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
