#!/usr/bin/env python3
"""
Step 4: Compute attack-curve labels for each window subgraph.

Primary Y (fixed 10 removal percentages, same for all window sizes):
  5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% -> relative efficiency loss
  (E0-E)/E0. No 100% (fully destroyed network is always ~0, not informative).

Per window, 30 targets (optional --also-auc adds 3 AUC scalars):
  - betweenness: deterministic, 1 run
  - capacity:    deterministic, 1 run
  - random:      n_random runs (default 10), mean at each percentage point

Pilot (--pilot-random): compare mean curves using first 10/20/30 of 30 shared
  permutations on a few windows; report max relative diff vs 30-rep mean.

Designed for batch runs on a remote machine (CPU multiprocessing).
Input: data/step4_gpu/manifest.csv + subgraphs/*.npz
        OR subgraphs.zip / subgraphs.tar.gz (auto-extract before compute)

Output:
  data/step4_gpu/labels/y_labels.csv
  data/step4_gpu/labels/y_labels.parquet
  data/step4_gpu/labels/curves/  (optional per-window curves)

Usage:
  python analysis/step4_compute_y_labels.py --workers 8
  python analysis/step4_pack_subgraphs.py
  python analysis/step4_compute_y_labels.py --archive data/step4_gpu/subgraphs.zip --workers 16
  python analysis/step4_compute_y_labels.py --data-dir data/step4_gpu --pilot-random
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from step4_archive_utils import extract_subgraphs_archive

# Fixed removal fractions (GNN output dim = 10 per attack; no 100%)
REMOVAL_PCTS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PCT_LABELS = ["p5", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]


def global_efficiency_weighted(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        total = 0.0
        count = 0
        for i in G.nodes():
            for j in G.nodes():
                if i == j:
                    continue
                d = lengths.get(i, {}).get(j, float("inf"))
                if np.isfinite(d) and d > 1e-12:
                    total += 1.0 / d
                count += 1
        return total / count if count else 0.0
    except Exception:
        return 0.0


def npz_to_graph(data: dict) -> nx.Graph:
    n = int(data["n_nodes"])
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, capacity=float(data["capacity"][i]))
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


def attack_curve(G: nx.Graph, order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return removal_frac[], relative_loss[] for k=1..n-1."""
    n = G.number_of_nodes()
    E0 = global_efficiency_weighted(G)
    if E0 <= 0 or n < 2:
        return np.array([]), np.array([])

    fracs = []
    losses = []
    for k in range(1, n):
        G2 = G.copy()
        G2.remove_nodes_from(order[:k])
        E = global_efficiency_weighted(G2)
        loss = max(0.0, (E0 - E) / E0)
        fracs.append(k / n)
        losses.append(loss)
    return np.asarray(fracs, dtype=np.float64), np.asarray(losses, dtype=np.float64)


def curve_auc(fracs: np.ndarray, losses: np.ndarray) -> float:
    if len(fracs) < 2:
        return float(losses[-1]) if len(losses) else 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(losses, fracs))
    return float(np.trapz(losses, fracs))  # NumPy < 2.0


def loss_at_removal_pcts(G: nx.Graph, order: list[int], pcts: list[float]) -> np.ndarray:
    """Relative efficiency loss at each target removal percentage."""
    n = G.number_of_nodes()
    E0 = global_efficiency_weighted(G)
    if E0 <= 0 or n < 2:
        return np.zeros(len(pcts), dtype=np.float64)

    # Incremental removal (one copy) instead of full G.copy() per percentage point.
    ks = [max(1, min(n - 1, int(round(p * n)))) for p in pcts]
    G2 = G.copy()
    losses = []
    prev_k = 0
    for k in ks:
        for i in range(prev_k, k):
            u = order[i]
            if G2.has_node(u):
                G2.remove_node(u)
        prev_k = k
        E = global_efficiency_weighted(G2)
        losses.append(max(0.0, (E0 - E) / E0))
    return np.asarray(losses, dtype=np.float64)


def _curve_to_columns(prefix: str, values: np.ndarray) -> dict:
    return {f"y_{prefix}_{lab}": float(values[i]) for i, lab in enumerate(PCT_LABELS)}


def random_curve_stack(
    G: nx.Graph,
    nodes: list,
    rng: np.random.Generator,
    n_random: int,
) -> np.ndarray:
    """Shape (n_random, len(REMOVAL_PCTS)) relative-loss curves."""
    stack = []
    for _ in range(n_random):
        perm = rng.permutation(nodes).tolist()
        stack.append(loss_at_removal_pcts(G, perm, REMOVAL_PCTS))
    return np.stack(stack, axis=0)


def process_one(
    subgraph_path: str,
    n_random: int = 10,
    random_seed: int = 42,
    save_curves: bool = False,
    curves_dir: str | None = None,
    also_auc: bool = False,
) -> dict:
    data = np.load(subgraph_path, allow_pickle=True)
    G = npz_to_graph(data)
    n = G.number_of_nodes()
    wid = str(data["window_id"]) if "window_id" in data else Path(subgraph_path).stem

    order_b = data["order_betweenness"].astype(int).tolist()
    order_c = data["order_capacity"].astype(int).tolist()

    lb = loss_at_removal_pcts(G, order_b, REMOVAL_PCTS)
    lc = loss_at_removal_pcts(G, order_c, REMOVAL_PCTS)

    rng = np.random.default_rng(random_seed + hash(wid) % (2**31))
    nodes = list(G.nodes())
    rand_stack = random_curve_stack(G, nodes, rng, n_random)
    lr = np.mean(rand_stack, axis=0)
    lr_std = np.std(rand_stack, axis=0) if n_random > 1 else np.zeros_like(lr)

    out = {
        "window_id": wid,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "e0_efficiency": global_efficiency_weighted(G),
    }
    out.update(_curve_to_columns("betweenness", lb))
    out.update(_curve_to_columns("capacity", lc))
    out.update(_curve_to_columns("random", lr))
    for i, lab in enumerate(PCT_LABELS):
        out[f"y_random_{lab}_std"] = float(lr_std[i])

    if also_auc:
        fb, lb_full = attack_curve(G, order_b)
        fc, lc_full = attack_curve(G, order_c)
        out["y_betweenness_auc"] = curve_auc(fb, lb_full)
        out["y_capacity_auc"] = curve_auc(fc, lc_full)
        rand_aucs = [
            curve_auc(*attack_curve(G, rng.permutation(nodes).tolist()))
            for _ in range(n_random)
        ]
        out["y_random_auc"] = float(np.mean(rand_aucs))
        out["y_random_auc_std"] = float(np.std(rand_aucs)) if len(rand_aucs) > 1 else 0.0

    if save_curves and curves_dir:
        cdir = Path(curves_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for attack, vec in (
            ("betweenness", lb),
            ("capacity", lc),
            ("random", lr),
        ):
            for p, val in zip(REMOVAL_PCTS, vec):
                rows.append({"attack": attack, "removal_pct": p, "relative_loss": val})
        pd.DataFrame(rows).to_csv(cdir / f"{wid}_curves.csv", index=False)

    return out


def _worker(args: tuple) -> dict:
    return process_one(*args)


def run_pilot_random_replicates(
    subgraph_paths: list[Path],
    *,
    compare_ns: tuple[int, ...] = (10, 20, 30),
    max_reps: int = 30,
    seed: int = 42,
    threshold: float = 0.01,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """
    On each window, draw max_reps random attack curves (same permutations).
    Compare mean of first k reps (k in compare_ns) to the max_reps mean.
    """
    max_reps = max(max_reps, max(compare_ns))
    rows = []
    for sp in subgraph_paths:
        data = np.load(sp, allow_pickle=True)
        G = npz_to_graph(data)
        wid = str(data["window_id"]) if "window_id" in data else sp.stem
        n = G.number_of_nodes()
        rng = np.random.default_rng(seed + hash(wid) % (2**31))
        nodes = list(G.nodes())
        stack = random_curve_stack(G, nodes, rng, max_reps)
        ref = np.mean(stack, axis=0)

        row = {"window_id": wid, "n_nodes": n, "n_edges": G.number_of_edges()}
        for k in compare_ns:
            if k > max_reps:
                continue
            mean_k = np.mean(stack[:k], axis=0)
            denom = np.maximum(np.abs(ref), 1e-9)
            rel = np.abs(mean_k - ref) / denom
            row[f"max_rel_diff_{k}_vs_{max_reps}"] = float(np.max(rel))
            row[f"mean_rel_diff_{k}_vs_{max_reps}"] = float(np.mean(rel))
            for lab, r in zip(PCT_LABELS, rel):
                row[f"rel_diff_{k}_vs_{max_reps}_{lab}"] = float(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote pilot report: {out_csv}")

    print("\n=== Pilot: random attack replicate count (vs 30-run mean) ===")
    for k in compare_ns:
        if k >= max_reps:
            continue
        col = f"max_rel_diff_{k}_vs_{max_reps}"
        if col not in df.columns:
            continue
        mx = float(df[col].max())
        mn = float(df[col].min())
        print(
            f"  n_random={k}: max rel diff across windows/points = {100 * mx:.3f}% "
            f"(min window {100 * mn:.3f}%)"
        )

    k10 = 10
    col10 = f"max_rel_diff_{k10}_vs_{max_reps}"
    if col10 in df.columns:
        ok = float(df[col10].max()) < threshold
        print(
            f"\n  Use n_random=10? max diff < {100 * threshold:.1f}% -> "
            f"{'YES' if ok else 'NO (consider more reps or check windows)'}"
        )
    return df


def _iter_with_progress(completed_iter, total: int, desc: str, *, enabled: bool = True):
    """Wrap as_completed() with tqdm (stderr) or periodic line prints."""
    if not enabled or total <= 0:
        yield from completed_iter
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    if tqdm is not None:
        with tqdm(
            completed_iter,
            total=total,
            desc=desc,
            unit="win",
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stderr,
            disable=False,
        ) as bar:
            for item in bar:
                yield item
        return

    print(f"{desc}: tqdm not installed (pip install tqdm); using line updates", flush=True)
    step = max(1, total // 100)
    t0 = time.perf_counter()
    n = 0
    for item in completed_iter:
        n += 1
        if n == 1 or n % step == 0 or n == total:
            elapsed = time.perf_counter() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            print(
                f"{desc}: {n}/{total} ({100.0 * n / total:.1f}%) "
                f"[{rate:.2f} win/s, {elapsed:.0f}s]",
                flush=True,
            )
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: compute 10-point attack curve labels")
    parser.add_argument("--data-dir", default="data/step4_gpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-random", type=int, default=10, help="Random attack replicates (mean curve)")
    parser.add_argument(
        "--pilot-random",
        action="store_true",
        help="Compare 10/20/30 random reps on pilot windows; exit without full batch",
    )
    parser.add_argument("--pilot-windows", type=int, default=5, help="Windows for --pilot-random")
    parser.add_argument("--pilot-max-reps", type=int, default=30, help="Reference rep count in pilot")
    parser.add_argument(
        "--pilot-threshold",
        type=float,
        default=0.01,
        help="Max relative diff (fraction) to recommend n_random=10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-curves", action="store_true")
    parser.add_argument("--also-auc", action="store_true", help="Also write y_*_auc columns")
    parser.add_argument(
        "--archive",
        default=None,
        help="subgraphs.zip or subgraphs.tar.gz; extract to data-dir/subgraphs/ before compute",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archive even if subgraphs/ already exists",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar / periodic status prints",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_dir)
    if data_path.is_absolute():
        data_dir = data_path.resolve()
    elif args.data_dir in (".", "./"):
        data_dir = Path.cwd().resolve()
    else:
        cand = (Path.cwd() / data_path).resolve()
        data_dir = cand if cand.exists() else (script_dir / data_path).resolve()

    if args.archive:
        archive = Path(args.archive)
        if not archive.is_absolute():
            archive = (data_dir / archive).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        print(f"Extracting {archive} -> {data_dir / 'subgraphs'} ...")
        extract_subgraphs_archive(archive, data_dir, force=args.force_extract)
    elif not (data_dir / "subgraphs").exists() or not any((data_dir / "subgraphs").glob("*.npz")):
        auto_zip = data_dir / "subgraphs.zip"
        auto_tgz = data_dir / "subgraphs.tar.gz"
        if auto_zip.exists():
            print(f"Found {auto_zip}, extracting...")
            extract_subgraphs_archive(auto_zip, data_dir)
        elif auto_tgz.exists():
            print(f"Found {auto_tgz}, extracting...")
            extract_subgraphs_archive(auto_tgz, data_dir)

    manifest = pd.read_csv(data_dir / "manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)

    if args.pilot_random:
        pilot_manifest = manifest.head(args.pilot_windows)
        paths = [data_dir / row["subgraph_path"] for _, row in pilot_manifest.iterrows()]
        if not paths:
            raise SystemExit("No subgraph paths for pilot (check manifest.csv)")
        print(f"Pilot random replicates on {len(paths)} windows (ref n={args.pilot_max_reps}) ...")
        run_pilot_random_replicates(
            paths,
            compare_ns=(10, 20, 30),
            max_reps=args.pilot_max_reps,
            seed=args.seed,
            threshold=args.pilot_threshold,
            out_csv=data_dir / "labels" / "pilot_random_replicates.csv",
        )
        return

    labels_dir = data_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = str(labels_dir / "curves") if args.save_curves else None

    tasks = []
    for _, row in manifest.iterrows():
        sp = data_dir / row["subgraph_path"]
        tasks.append((str(sp), args.n_random, args.seed, args.save_curves, curves_dir, args.also_auc))

    n_tasks = len(tasks)
    print(
        f"Computing labels for {n_tasks} windows "
        f"({args.workers} workers, n_random={args.n_random}) ...",
        flush=True,
    )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in _iter_with_progress(
            as_completed(futures),
            total=n_tasks,
            desc="Step4 Y labels",
            enabled=not args.no_progress,
        ):
            rows.append(fut.result())

    df = pd.DataFrame(rows).merge(manifest, on="window_id", how="left")
    out_csv = labels_dir / "y_labels.csv"
    out_parquet = labels_dir / "y_labels.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)

    print(f"Wrote {out_csv} ({len(df)} windows)")
    print(f"Label columns per attack: {PCT_LABELS} (30 cols + optional AUC)")
    demo_cols = [f"y_betweenness_{PCT_LABELS[0]}", f"y_betweenness_{PCT_LABELS[-1]}"]
    print(df[demo_cols].describe())


if __name__ == "__main__":
    main()
