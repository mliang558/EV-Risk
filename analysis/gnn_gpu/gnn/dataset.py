"""Build PyG Data objects from Step-4 subgraphs + labels."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

_ANALYSIS = Path(__file__).resolve().parents[1]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from gnn.constants import Y_ALL_COLS
from gnn.edge_utils import ensure_graph_level_y, npz_to_pyg_edges, sanitize_graph_list
from gnn.node_features import (
    build_dcfc_lookup,
    features_from_npz,
    node_feature_matrix,
    window_graph_from_nodes,
)


_MP_DCFC: pd.DataFrame | None = None
_MP_NETWORK_DIR: str | None = None
_MP_NET_CACHE: dict = {}


def _mp_init(dcfc_parquet: str | None, network_dir: str | None) -> None:
    global _MP_DCFC, _MP_NETWORK_DIR, _MP_NET_CACHE
    _MP_DCFC = pd.read_parquet(dcfc_parquet) if dcfc_parquet and Path(dcfc_parquet).exists() else None
    _MP_NETWORK_DIR = network_dir
    _MP_NET_CACHE = {}


def _build_one_window(task: tuple) -> tuple | None:
    """Worker: return (index, raw_x, edge_index, edge_attr, y, meta_dict, num_nodes) or None."""
    idx, row, data_dir_str, wn_records = task
    data_dir = Path(data_dir_str)
    wid = row["window_id"]
    sp = data_dir / row.get("subgraph_path", f"subgraphs/{wid}.npz")
    if not sp.exists():
        sp = data_dir / "subgraphs" / f"{wid}.npz"
    if not sp.exists():
        return None

    npz = np.load(sp, allow_pickle=True)
    ei = npz["edge_index"]
    ew = npz["edge_weight"]
    x = features_from_npz(npz)
    if x is None:
        if not wn_records or not _MP_NETWORK_DIR:
            return None
        wn = pd.DataFrame(wn_records)
        G, old_keys = window_graph_from_nodes(wn, Path(_MP_NETWORK_DIR), _MP_NET_CACHE)
        x = node_feature_matrix(G, old_keys, _MP_DCFC)

    y = np.array([float(row[c]) for c in Y_ALL_COLS], dtype=np.float32)
    meta = {
        "window_id": wid,
        "center_lat": float(row.get("center_lat", 0.0)),
        "center_lon": float(row.get("center_lon", 0.0)),
        "states": str(row.get("states", "")),
        "window_stratum": str(row.get("window_stratum", "")),
    }
    return (
        idx,
        x.astype(np.float32),
        ei.astype(np.int64),
        ew.astype(np.float32),
        y,
        meta,
        int(x.shape[0]),
    )


def standardize_features(feat_list: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    stack = np.vstack(feat_list)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    std[std < 1e-8] = 1.0
    out = [(f - mean) / std for f in feat_list]
    return out, mean, std


def build_dataset(
    data_dir: Path,
    *,
    labels_path: Path | None = None,
    windows_nodes_path: Path | None = None,
    network_dir: Path | None = None,
    stations_csv: Path | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[list[Data], pd.DataFrame, dict]:
    """
    Returns (list of Data, labels_df merged with meta, stats dict).
    """
    data_dir = data_dir.resolve()
    print(f"[Step5] Loading tables from {data_dir} ...", flush=True)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    meta_path = data_dir / "windows_meta.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else manifest

    if labels_path is None:
        labels_path = data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)

    if "subgraph_path" not in labels.columns:
        labels = labels.merge(manifest[["window_id", "subgraph_path"]], on="window_id", how="left")
    df = labels.merge(meta, on="window_id", how="inner", suffixes=("", "_meta"))
    if limit:
        df = df.head(limit)
    print(f"[Step5] {len(df)} windows to build.", flush=True)

    wnodes = None
    if windows_nodes_path and Path(windows_nodes_path).exists():
        print(f"[Step5] Loading {windows_nodes_path} ...", flush=True)
        wnodes = pd.read_parquet(windows_nodes_path)

    dcfc_cache = data_dir / "dcfc_lookup.parquet"
    if dcfc_cache.exists():
        print(f"[Step5] Loading cached DCFC lookup: {dcfc_cache}", flush=True)
        dcfc_lookup = pd.read_parquet(dcfc_cache)
    elif stations_csv and Path(stations_csv).exists():
        print("Building DCFC lookup from stations CSV (once, ~1–2 min)...", flush=True)
        dcfc_lookup = build_dcfc_lookup(Path(stations_csv))
        dcfc_lookup.to_parquet(dcfc_cache, index=False)
        print(f"[Step5] Cached DCFC -> {dcfc_cache}", flush=True)
    else:
        dcfc_lookup = None

    wnodes_by_wid: dict[str, pd.DataFrame] = {}
    if wnodes is not None:
        print("Indexing windows_nodes by window_id...", flush=True)
        wnodes_by_wid = {wid: g for wid, g in wnodes.groupby("window_id", sort=False)}

    # Fast path: npz already has lat/lon/dcfc (after enrich) — minutes instead of ~1h.
    sample_npz = data_dir / str(df.iloc[0].get("subgraph_path", f"subgraphs/{df.iloc[0]['window_id']}.npz"))
    if not sample_npz.exists():
        sample_npz = data_dir / "subgraphs" / f"{df.iloc[0]['window_id']}.npz"
    npz_fast = False
    if sample_npz.exists():
        with np.load(sample_npz, allow_pickle=True) as z:
            npz_fast = features_from_npz(z) is not None
    if npz_fast:
        print("[Step5] npz already has lat/lon/dcfc — fast build (no network pickle).", flush=True)

    graphs: list[Data | None] = [None] * len(df)
    raw_feats: list[np.ndarray | None] = [None] * len(df)
    meta_rows: list[dict | None] = [None] * len(df)
    skipped = 0
    n_df = len(df)

    dcfc_path = str(dcfc_cache) if dcfc_cache.exists() else None
    if network_dir is None:
        try:
            from network_paths import resolve_network_dir

            network_dir = resolve_network_dir(cwd=data_dir)
            print(f"[Step5] Auto network_dir={network_dir}", flush=True)
        except FileNotFoundError:
            network_dir = None
    net_dir_str = str(network_dir) if network_dir else None

    if workers > 1 and not npz_fast and not net_dir_str:
        print(
            "[Step5] npz lacks lat/lon/dcfc — need --network-dir network_structures "
            "(or run with --workers 1). Forcing workers=1.",
            flush=True,
        )
        workers = 1

    if workers > 1 and not npz_fast:
        print(f"[Step5] CPU parallel build with {workers} workers (GPU not used here).", flush=True)
        tasks = []
        for i in range(n_df):
            row = df.iloc[i].to_dict()
            wid = row["window_id"]
            wn_rec = None
            if wid in wnodes_by_wid:
                wn_rec = wnodes_by_wid[wid].to_dict("records")
            tasks.append((i, row, str(data_dir), wn_rec))

        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_mp_init,
            initargs=(dcfc_path, net_dir_str),
        ) as pool:
            futures = [pool.submit(_build_one_window, t) for t in tasks]
            it = as_completed(futures)
            if tqdm is not None:
                it = tqdm(it, total=len(futures), desc="Step5 PyG graphs", unit="win", file=sys.stderr)
            for fut in it:
                res = fut.result()
                if res is None:
                    skipped += 1
                    continue
                idx, x, ei_np, ew_np, y, meta, n_nodes = res
                edge_index, edge_attr = npz_to_pyg_edges(ei_np, ew_np)
                y_tensor = torch.from_numpy(y).view(1, -1)
                raw_feats[idx] = x
                graphs[idx] = Data(
                    x=torch.from_numpy(x.copy()),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y_tensor,
                    y_betweenness=y_tensor[:, :10].clone(),
                    y_capacity=y_tensor[:, 10:20].clone(),
                    y_random=y_tensor[:, 20:30].clone(),
                    num_nodes=n_nodes,
                )
                graphs[idx].window_id = meta["window_id"]
                graphs[idx].center_lat = meta["center_lat"]
                graphs[idx].center_lon = meta["center_lon"]
                graphs[idx].states = meta["states"]
                graphs[idx].window_stratum = meta["window_stratum"]
                meta_rows[idx] = {**df.iloc[idx].to_dict(), **meta}
                done += 1
        print(f"[Step5] Built {done} graphs ({skipped} skipped).", flush=True)
    else:
        net_cache: dict = {}
        use_tqdm = tqdm is not None
        if use_tqdm:
            pbar = tqdm(total=n_df, desc="Step5 PyG graphs", unit="win", file=sys.stderr, mininterval=0.5)
        t0 = time.perf_counter()
        log_step = max(1, n_df // 50)
        built = 0

        for i in range(n_df):
            row = df.iloc[i]
            if use_tqdm:
                pbar.update(1)
            elif (i + 1) % log_step == 0 or i == 0 or i + 1 == n_df:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"[Step5] {i + 1}/{n_df} ({100 * (i + 1) / n_df:.1f}%) {rate:.1f} win/s",
                    flush=True,
                )
            wid = row["window_id"]
            sp = data_dir / row["subgraph_path"]
            if not sp.exists():
                sp = data_dir / "subgraphs" / f"{wid}.npz"
            if not sp.exists():
                skipped += 1
                continue

            npz = np.load(sp, allow_pickle=True)
            ei_np = npz["edge_index"]
            ew_np = npz["edge_weight"]
            edge_index, edge_attr = npz_to_pyg_edges(ei_np, ew_np)

            x = features_from_npz(npz)
            if x is None:
                if wnodes_by_wid and network_dir is not None:
                    wn = wnodes_by_wid.get(wid)
                    if wn is None or wn.empty:
                        skipped += 1
                        continue
                    G, old_keys = window_graph_from_nodes(wn, Path(network_dir), net_cache)
                    x = node_feature_matrix(G, old_keys, dcfc_lookup)
                else:
                    raise FileNotFoundError(
                        f"{wid}: need lat/lon in npz OR --windows-nodes + --network-dir"
                    )

            y = row[Y_ALL_COLS].astype(float).values.astype(np.float32)
            y_tensor = torch.from_numpy(y).view(1, -1)
            raw_feats[i] = x
            graphs[i] = Data(
                x=torch.from_numpy(x.copy()),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_tensor,
                y_betweenness=y_tensor[:, :10].clone(),
                y_capacity=y_tensor[:, 10:20].clone(),
                y_random=y_tensor[:, 20:30].clone(),
                num_nodes=int(x.shape[0]),
            )
            graphs[i].window_id = wid
            graphs[i].center_lat = float(row.get("center_lat", 0.0))
            graphs[i].center_lon = float(row.get("center_lon", 0.0))
            graphs[i].states = str(row.get("states", ""))
            graphs[i].window_stratum = str(row.get("window_stratum", ""))
            meta_rows[i] = row.to_dict()
            built += 1

        if use_tqdm:
            pbar.close()
        print(f"[Step5] Built {built} graphs ({skipped} skipped).", flush=True)

    graphs = [g for g in graphs if g is not None]
    raw_feats = [f for f in raw_feats if f is not None]
    meta_rows = [m for m in meta_rows if m is not None]

    if not graphs:
        raise RuntimeError(
            "No graphs built. npz files lack lat/lon/dcfc — re-run with:\n"
            "  python step5_build_pyg_dataset.py --data-dir . "
            "--network-dir network_structures --workers 16"
        )

    print(f"[Step5] Standardizing features ({len(raw_feats)} graphs)...", flush=True)
    normed, mean, std = standardize_features(raw_feats)
    for g, xn in zip(graphs, normed):
        g.x = torch.from_numpy(xn.astype(np.float32))
    n_fix = sanitize_graph_list(graphs)
    if n_fix:
        print(f"[Step5] Sanitized edges/num_nodes on {n_fix} graphs.", flush=True)

    stats = {
        "n_graphs": len(graphs),
        "skipped": skipped,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "y_dim": 30,
    }
    meta_out = pd.DataFrame(meta_rows)
    return graphs, meta_out, stats


def save_dataset_bundle(
    out_dir: Path,
    graphs: list[Data],
    meta_df: pd.DataFrame,
    stats: dict,
    split_masks: dict[str, np.ndarray] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "graphs": graphs,
        "meta": meta_df,
        "stats": stats,
        "split_masks": split_masks or {},
    }
    path = out_dir / "pyg_dataset.pt"
    torch.save(bundle, path)
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    meta_df.to_csv(out_dir / "dataset_meta.csv", index=False)
    return path
