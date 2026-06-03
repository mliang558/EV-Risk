"""Per-window global graph statistics (concatenated to GNN readout)."""

from __future__ import annotations

from pathlib import Path

import sys

import networkx as nx
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Original 15-dim set (LightGBM / GAT+global baseline)
GLOBAL_FEATURE_NAMES_BASE = (
    "n_nodes",
    "n_edges",
    "density",
    "total_capacity",
    "mean_capacity",
    "std_capacity",
    "max_capacity",
    "avg_degree",
    "max_degree",
    "avg_clustering",
    "max_betweenness",
    "mean_betweenness",
    "global_efficiency",
    "diameter",
    "n_components",
)

# Extra topology for betweenness-style attacks (Fiedler, paths, degree spread)
GLOBAL_FEATURE_NAMES_EXTRA = (
    "algebraic_connectivity",
    "avg_shortest_path",
    "degree_heterogeneity",
)

GLOBAL_FEATURE_NAMES_EXTENDED = GLOBAL_FEATURE_NAMES_BASE + GLOBAL_FEATURE_NAMES_EXTRA

# Backward-compatible alias
GLOBAL_FEATURE_NAMES = GLOBAL_FEATURE_NAMES_BASE


def get_global_feature_names(feature_set: str = "base") -> tuple[str, ...]:
    """base=15 | extended=18 (+ Fiedler, mean shortest path, degree std/mean)."""
    s = str(feature_set).strip().lower()
    if s in ("base", "default", "15", "legacy"):
        return GLOBAL_FEATURE_NAMES_BASE
    if s in ("extended", "ext", "18", "betweenness", "full"):
        return GLOBAL_FEATURE_NAMES_EXTENDED
    raise ValueError(f"Unknown feature_set {feature_set!r}; use base | extended")


def npz_to_graph(data) -> nx.Graph:
    n = int(data["n_nodes"])
    G = nx.Graph()
    caps = data["capacity"]
    for i in range(n):
        G.add_node(i, capacity=float(caps[i]))
    ei = data["edge_index"]
    w = data["edge_weight"]
    seen = set()
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        G.add_edge(u, v, weight=float(w[e]))
    return G


def _topology_extra(G: nx.Graph) -> dict[str, float]:
    """Features targeted at betweenness / robustness (requires NetworkX)."""
    n = G.number_of_nodes()
    out = {
        "algebraic_connectivity": 0.0,
        "avg_shortest_path": 0.0,
        "degree_heterogeneity": 0.0,
    }
    if n < 2 or G.number_of_edges() == 0:
        return out

    degrees = np.array([float(d) for _, d in G.degree()], dtype=np.float64)
    mean_d = float(degrees.mean())
    out["degree_heterogeneity"] = float(degrees.std() / mean_d) if mean_d > 1e-8 else 0.0

    try:
        out["algebraic_connectivity"] = float(nx.algebraic_connectivity(G, weight="weight"))
    except Exception:
        try:
            out["algebraic_connectivity"] = float(nx.algebraic_connectivity(G))
        except Exception:
            pass

    try:
        if nx.is_connected(G):
            out["avg_shortest_path"] = float(nx.average_shortest_path_length(G, weight="weight"))
        else:
            comps = [c for c in nx.connected_components(G) if len(c) > 1]
            if comps:
                largest = max(comps, key=len)
                sg = G.subgraph(largest)
                out["avg_shortest_path"] = float(
                    nx.average_shortest_path_length(sg, weight="weight")
                )
    except Exception:
        pass
    return out


def global_features_from_npz(
    npz_path: Path,
    *,
    fast: bool = False,
    feature_set: str = "base",
) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    names = get_global_feature_names(feature_set)
    if fast:
        fd = _global_features_from_npz_fast(data, feature_set=feature_set)
        for k in names:
            fd.setdefault(k, 0.0)
        return {k: fd[k] for k in names}

    G = npz_to_graph(data)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    caps = np.array([float(G.nodes[i].get("capacity", 0.0)) for i in G.nodes()], dtype=float)

    feats: dict[str, float] = {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "total_capacity": float(caps.sum()),
        "mean_capacity": float(caps.mean()) if n else 0.0,
        "std_capacity": float(caps.std()) if n > 1 else 0.0,
        "max_capacity": float(caps.max()) if n else 0.0,
    }

    if n >= 2 and m > 0:
        feats["avg_degree"] = float(2 * m / n)
        feats["max_degree"] = float(max(d for _, d in G.degree()))
        try:
            feats["avg_clustering"] = float(nx.average_clustering(G, weight="weight"))
        except Exception:
            feats["avg_clustering"] = 0.0
        try:
            b = nx.betweenness_centrality(G, weight="weight")
            feats["max_betweenness"] = float(max(b.values()))
            feats["mean_betweenness"] = float(np.mean(list(b.values())))
        except Exception:
            feats["max_betweenness"] = 0.0
            feats["mean_betweenness"] = 0.0
        try:
            feats["global_efficiency"] = float(nx.global_efficiency(G))
        except Exception:
            feats["global_efficiency"] = 0.0
        try:
            if nx.is_connected(G):
                feats["diameter"] = float(nx.diameter(G))
            else:
                comps = [c for c in nx.connected_components(G) if len(c) > 1]
                feats["diameter"] = float(max(nx.diameter(G.subgraph(c)) for c in comps)) if comps else 0.0
        except Exception:
            feats["diameter"] = 0.0
        feats["n_components"] = float(nx.number_connected_components(G))
    else:
        for k in GLOBAL_FEATURE_NAMES_BASE[8:]:
            feats[k] = 0.0

    if feature_set.lower() in ("extended", "ext", "18", "betweenness", "full"):
        feats.update(_topology_extra(G))

    return {k: float(feats.get(k, 0.0)) for k in names}


def _global_features_from_npz_fast(data, feature_set: str = "base") -> dict[str, float]:
    """No NetworkX betweenness — avoids OOM / bus error on large windows."""
    n = int(data["n_nodes"])
    caps = np.asarray(data["capacity"], dtype=np.float64)
    ei = data["edge_index"]
    m = float(max(0, ei.shape[1] // 2))
    feats: dict[str, float] = {
        "n_nodes": float(n),
        "n_edges": m,
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "total_capacity": float(caps.sum()),
        "mean_capacity": float(caps.mean()) if n else 0.0,
        "std_capacity": float(caps.std()) if n > 1 else 0.0,
        "max_capacity": float(caps.max()) if n else 0.0,
        "avg_degree": (2.0 * m / n) if n else 0.0,
        "max_degree": 0.0,
        "avg_clustering": 0.0,
        "max_betweenness": 0.0,
        "mean_betweenness": 0.0,
        "global_efficiency": 0.0,
        "diameter": 0.0,
        "n_components": 1.0,
    }
    if n > 0 and m > 0:
        deg = np.zeros(n, dtype=np.float64)
        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            if u == v:
                continue
            deg[u] += 1
            deg[v] += 1
        feats["max_degree"] = float(deg.max())
        mean_d = float(deg.mean())
        feats["degree_heterogeneity"] = float(deg.std() / mean_d) if mean_d > 1e-8 else 0.0
    names = get_global_feature_names(feature_set)
    for k in names:
        feats.setdefault(k, 0.0)
    return {k: feats[k] for k in names}


def build_window_global_features_table(
    data_dir: Path,
    window_ids: list[str],
    *,
    feature_set: str = "base",
    fast: bool = False,
) -> "pd.DataFrame":
    """One row per window_id with global features (for CSV cache / LGB)."""
    import pandas as pd

    names = get_global_feature_names(feature_set)
    rows = []
    it = window_ids
    if tqdm is not None:
        it = tqdm(window_ids, desc=f"Global feats ({feature_set})", unit="win", file=sys.stderr)
    for wid in it:
        sp = subgraph_path_for_window(data_dir, str(wid))
        if sp is None:
            continue
        fd = global_features_from_npz(sp, fast=fast, feature_set=feature_set)
        rows.append({"window_id": str(wid), **{k: fd[k] for k in names}})
    return pd.DataFrame(rows)


def subgraph_path_for_window(data_dir: Path, window_id: str) -> Path | None:
    data_dir = data_dir.resolve()
    wid = str(window_id)
    for cand in (
        data_dir / "subgraphs" / f"{wid}.npz",
        data_dir / f"subgraphs/{wid}.npz",
    ):
        if cand.exists():
            return cand
    return None


def attach_global_features_from_csv(
    graphs: list,
    csv_path: Path,
    train_mask: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    verbose: bool = True,
) -> dict:
    """Attach gf from precomputed window_global_features.csv (same as LightGBM run)."""
    import pandas as pd

    feat_df = pd.read_csv(csv_path)
    if "window_id" not in feat_df.columns:
        raise ValueError(f"{csv_path} needs column window_id")
    want = feature_names or GLOBAL_FEATURE_NAMES_BASE
    cols = [c for c in want if c in feat_df.columns]
    if not cols:
        cols = [c for c in feat_df.columns if c != "window_id"]
    if not cols:
        raise ValueError(f"No feature columns in {csv_path}")

    by_wid = feat_df.set_index("window_id")
    raw = np.zeros((len(graphs), len(cols)), dtype=np.float64)
    missing = 0
    for i, g in enumerate(graphs):
        wid = str(getattr(g, "window_id", ""))
        if wid not in by_wid.index:
            missing += 1
            continue
        raw[i] = by_wid.loc[wid, cols].astype(float).values

    tr = np.asarray(train_mask, dtype=bool)
    mean = raw[tr].mean(axis=0) if tr.sum() >= 2 else raw.mean(axis=0)
    std = raw[tr].std(axis=0) if tr.sum() >= 2 else raw.std(axis=0)
    std[std < 1e-8] = 1.0
    normed = (raw - mean) / std
    for i, g in enumerate(graphs):
        g.gf = torch.from_numpy(normed[i].astype(np.float32))

    if verbose:
        print(f"[global_feat] from CSV {csv_path} | F={len(cols)} | missing={missing}", flush=True)
    return {
        "global_feat_dim": len(cols),
        "global_feat_names": cols,
        "global_feat_mean": mean.tolist(),
        "global_feat_std": std.tolist(),
        "global_feat_source": str(csv_path),
    }


def attach_global_features_to_graphs(
    graphs: list,
    data_dir: Path,
    train_mask: np.ndarray,
    *,
    fast: bool = False,
    feature_set: str = "base",
    verbose: bool = True,
) -> dict:
    """
    Set graph.gf (float32 tensor, shape [F]) on each Data; standardize using train split.
  """
    data_dir = data_dir.resolve()
    names = get_global_feature_names(feature_set)
    raw = np.zeros((len(graphs), len(names)), dtype=np.float64)
    missing = 0
    it = enumerate(graphs)
    if verbose and tqdm is not None:
        it = tqdm(it, total=len(graphs), desc="Global features", unit="win", file=sys.stderr)
    for i, g in it:
        wid = str(getattr(g, "window_id", f"idx{i}"))
        sp = subgraph_path_for_window(data_dir, wid)
        if sp is None:
            missing += 1
            continue
        fd = global_features_from_npz(sp, fast=fast, feature_set=feature_set)
        raw[i] = [fd[k] for k in names]

    if missing and verbose:
        print(f"[global_feat] missing npz for {missing} graphs", flush=True)

    tr = np.asarray(train_mask, dtype=bool)
    if tr.sum() < 2:
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
    else:
        mean = raw[tr].mean(axis=0)
        std = raw[tr].std(axis=0)
    std[std < 1e-8] = 1.0
    normed = (raw - mean) / std

    for i, g in enumerate(graphs):
        g.gf = torch.from_numpy(normed[i].astype(np.float32))

    stats = {
        "global_feat_dim": len(names),
        "global_feat_names": list(names),
        "global_feat_set": feature_set,
        "global_feat_mean": mean.tolist(),
        "global_feat_std": std.tolist(),
    }
    if verbose:
        print(
            f"[global_feat] attached F={len(names)} ({feature_set}) to {len(graphs)} graphs",
            flush=True,
        )
    return stats


def strip_global_features(graphs: list) -> None:
    for g in graphs:
        if hasattr(g, "gf"):
            delattr(g, "gf")


def _graph_global_dim(graphs: list) -> int | None:
    if not graphs:
        return None
    gf = getattr(graphs[0], "gf", None)
    if gf is None:
        return None
    return int(gf.numel())


def global_feature_names_from_ckpt(ckpt: dict | None) -> tuple[str, ...] | None:
    if not ckpt:
        return None
    stats = ckpt.get("stats") or {}
    raw = stats.get("global_feat_names") or ckpt.get("global_feat_names")
    if raw:
        return tuple(raw)
    fs = ckpt.get("global_feature_set")
    if fs:
        return get_global_feature_names(fs)
    return None


def resolve_global_features_csv(
    data_dir: Path,
    *,
    csv_path: Path | str | None = None,
    feature_set: str = "extended",
) -> Path | None:
    data_dir = data_dir.resolve()
    if csv_path is not None:
        p = Path(csv_path)
        if p.is_file():
            return p
    for cand in (
        data_dir / f"results_gnn/global_{feature_set}/window_global_features.csv",
        data_dir / "results_gnn/_smoke/window_global_features.csv",
        data_dir / "results_gnn/global_extended/window_global_features.csv",
        data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
        data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
    ):
        if cand.is_file():
            return cand
    return None


def ensure_global_features_for_eval(
    graphs: list,
    global_dim: int,
    train_mask: np.ndarray,
    data_dir: Path,
    *,
    ckpt: dict | None = None,
    csv_path: Path | str | None = None,
    force_reattach: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Align graph.gf with model global_dim. Strips stale gf (e.g. 15-dim in pyg_dataset.pt)
    and re-attaches from CSV or npz when dimensions differ or reattach is requested.
    """
    if global_dim <= 0:
        strip_global_features(graphs)
        return {}

    data_dir = data_dir.resolve()
    feature_set = str((ckpt or {}).get("global_feature_set", "")).strip().lower()
    if not feature_set:
        feature_set = "extended" if global_dim > len(GLOBAL_FEATURE_NAMES_BASE) else "base"

    feat_names = global_feature_names_from_ckpt(ckpt)
    if feat_names is None or len(feat_names) != global_dim:
        feat_names = get_global_feature_names(feature_set)

    explicit_csv = Path(csv_path) if csv_path else None
    if explicit_csv is not None and not explicit_csv.is_file():
        explicit_csv = None

    current = _graph_global_dim(graphs)
    need = (
        force_reattach
        or current is None
        or current != global_dim
        or explicit_csv is not None
    )
    if not need:
        return {}

    strip_global_features(graphs)
    resolved = explicit_csv or resolve_global_features_csv(
        data_dir, feature_set=feature_set
    )
    if resolved is not None:
        return attach_global_features_from_csv(
            graphs,
            resolved,
            train_mask,
            feature_names=feat_names,
            verbose=verbose,
        )
    return attach_global_features_to_graphs(
        graphs,
        data_dir,
        train_mask,
        fast=True,
        feature_set=feature_set,
        verbose=verbose,
    )
