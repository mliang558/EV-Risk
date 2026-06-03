#!/usr/bin/env python3
"""
Benchmark simulation (Step 4) vs GNN inference (Step 6) for paper speed claims.

Records:
  - Training: GPU name, wall time, epochs (from train_meta.json / history.json / checkpoint)
  - Simulation: seconds per graph (full attack-curve labels, CPU)
  - Inference: milliseconds per graph (GAT forward pass, GPU)
  - Speedup: simulation_sec / inference_sec

Usage (after Step 5 + Step 6):
  cd /opt/data_repo/mliang_work/step4_gpu
  python step8_benchmark_speed.py \\
    --dataset pyg/pyg_dataset.pt \\
    --checkpoint pyg/train_gat/best_model.pt \\
    --data-dir . \\
    --device cuda \\
    --n-samples 100

Output:
  results_gnn/benchmark/speed_benchmark.json
  results_gnn/benchmark/speed_benchmark.md   (paper-ready table)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.edge_utils import (
    edge_attr_row_count,
    ensure_graph_level_y,
    force_sync_edge_attr,
    reload_edges_from_npz,
    sanitize_graph_edges,
    sanitize_graph_list,
)

STEP8_VERSION = "2026-05-31-edge-fix-v2"
from gnn.mlflow_utils import MlflowTracker
from gnn.models import ResilienceGNN


def load_step4_process_one():
    step4_path = _ROOT / "step4_compute_y_labels.py"
    if not step4_path.exists():
        step4_path = Path.cwd() / "step4_compute_y_labels.py"
    spec = importlib.util.spec_from_file_location("step4_labels", step4_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.process_one


def gpu_info(device: torch.device) -> dict:
    out = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda" and torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        out["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return out


def load_training_meta(train_dir: Path, ckpt: dict) -> dict:
    meta = {}
    meta_path = train_dir / "train_meta.json"
    hist_path = train_dir / "history.json"
    if meta_path.exists():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        meta["epochs_ran"] = len(hist)
        if hist:
            meta["best_epoch"] = int(ckpt.get("epoch", hist[-1].get("epoch", len(hist))))
            meta["best_test_mae"] = hist[-1].get("test_mae")
            for row in reversed(hist):
                if "test_mae" in row:
                    meta["final_test_mae"] = row["test_mae"]
                    break
    meta.setdefault("best_epoch", int(ckpt.get("epoch", 0)))
    meta.setdefault("model_type", ckpt.get("model_type", "gat"))
    return meta


def load_model(ckpt_path: Path, device: torch.device) -> ResilienceGNN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=int(ckpt.get("hidden", 32)),
        gat_heads=int(ckpt.get("gat_heads", 4)),
        edge_dim=1 if ckpt.get("model_type", "gat") == "gat" else 0,
        model_type=ckpt.get("model_type", "gat"),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def subgraph_path_for_graph(data_dir: Path, g, meta_row: dict | None) -> Path:
    wid = getattr(g, "window_id", None)
    if meta_row and "subgraph_path" in meta_row:
        p = data_dir / meta_row["subgraph_path"]
        if p.exists():
            return p
    if wid:
        p = data_dir / "subgraphs" / f"{wid}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"No subgraph npz for window_id={wid}")


def prepare_graph_for_inference(data, npz_path: Path | None = None) -> None:
    """Align edges before GAT; prefer npz rebuild (matches Step 5)."""
    if npz_path is not None and npz_path.exists():
        reload_edges_from_npz(data, npz_path)
    else:
        sanitize_graph_edges(data)
    force_sync_edge_attr(data)
    ensure_graph_level_y(data)
    ne = int(data.edge_index.size(1)) if data.edge_index is not None else 0
    na = edge_attr_row_count(data)
    if ne != na:
        raise ValueError(f"edge_index E={ne} != edge_attr rows {na} after prepare")


@torch.no_grad()
def time_inference_ms(
    model,
    data,
    device: torch.device,
    npz_path: Path | None = None,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    prepare_graph_for_inference(data, npz_path=npz_path)
    batch = Batch.from_data_list([data]).to(device)
    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def time_simulation_sec(process_one, npz_path: str, n_random: int, repeats: int = 1) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        process_one(npz_path, n_random=n_random)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simulation vs GNN inference")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", default="pyg/train_gat/best_model.pt")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--train-dir", default=None, help="Dir with history.json (default: checkpoint parent)")
    parser.add_argument("--out", default="results_gnn/benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=10, help="Match Step 4 n_random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-repeats", type=int, default=1, help="Simulation runs per graph (1=typical)")
    parser.add_argument("--inf-warmup", type=int, default=5)
    parser.add_argument("--inf-repeats", type=int, default=20)
    parser.add_argument(
        "--inf-only",
        action="store_true",
        help="Skip Step 4 simulation (only benchmark GNN inference latency)",
    )
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", default=True)
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    parser.add_argument("--mlflow-experiment", default="ev-charging-resilience-gnn")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = torch.load(Path(args.dataset).resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(Path(args.dataset).resolve(), map_location="cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs)
    meta_df = bundle.get("meta")
    meta_by_wid = {}
    if meta_df is not None:
        for _, r in meta_df.iterrows():
            meta_by_wid[str(r["window_id"])] = r.to_dict()

    ckpt_path = Path(args.checkpoint).resolve()
    model, ckpt = load_model(ckpt_path, device)
    train_dir = Path(args.train_dir).resolve() if args.train_dir else ckpt_path.parent
    training = load_training_meta(train_dir, ckpt)
    training["gpu"] = gpu_info(device)

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(graphs))
    indices = rng.choice(len(graphs), size=n, replace=False)

    process_one = load_step4_process_one()
    sim_times: list[float] = []
    inf_times: list[float] = []
    nodes_list: list[int] = []
    per_graph: list[dict] = []

    print(
        f"Benchmarking {n} graphs | device={device} | n_random={args.n_random} | {STEP8_VERSION}",
        flush=True,
    )

    for k, idx in enumerate(indices):
        g = graphs[int(idx)]
        wid = str(getattr(g, "window_id", idx))
        meta_row = meta_by_wid.get(wid)
        sp = subgraph_path_for_graph(data_dir, g, meta_row)

        try:
            inf_ms = time_inference_ms(
                model,
                g,
                device,
                npz_path=sp,
                warmup=args.inf_warmup,
                repeats=args.inf_repeats,
            )
        except Exception as e:
            print(f"[Step8] Skip inference {wid}: {e}", flush=True)
            continue

        if args.inf_only:
            sim_s = float("nan")
        else:
            sim_s = time_simulation_sec(process_one, str(sp), args.n_random, repeats=args.sim_repeats)

        sim_times.append(sim_s)
        inf_times.append(inf_ms)
        nodes_list.append(int(g.num_nodes))
        per_graph.append(
            {
                "window_id": wid,
                "n_nodes": int(g.num_nodes),
                "simulation_sec": sim_s,
                "inference_ms": inf_ms,
                "speedup_x": sim_s / (inf_ms / 1000.0),
            }
        )
        if (k + 1) % max(1, n // 10) == 0 or k + 1 == n:
            print(f"  {k + 1}/{n} | sim={sim_s:.3f}s | inf={inf_ms:.2f}ms", flush=True)

    if not inf_times:
        raise RuntimeError("No successful inference runs; upload step8 + gnn/edge_utils.py (see STEP8_VERSION)")

    sim_med = float(np.nanmedian(sim_times)) if not args.inf_only else float("nan")
    sim_mean = float(np.nanmean(sim_times)) if not args.inf_only else float("nan")
    inf_med = float(np.median(inf_times))
    inf_mean = float(np.mean(inf_times))
    speedup_med = sim_med / (inf_med / 1000.0)
    speedup_mean = sim_mean / (inf_mean / 1000.0)

    report = {
        "training": training,
        "benchmark": {
            "n_graphs_sampled": n,
            "n_random_simulation": args.n_random,
            "simulation_sec_median": sim_med,
            "simulation_sec_mean": sim_mean,
            "inference_ms_median": inf_med,
            "inference_ms_mean": inf_mean,
            "speedup_median_x": speedup_med,
            "speedup_mean_x": speedup_mean,
            "n_nodes_median": float(np.median(nodes_list)),
            "n_nodes_mean": float(np.mean(nodes_list)),
        },
        "per_graph": per_graph,
    }

    json_path = out_dir / "speed_benchmark.json"
    md_path = out_dir / "speed_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    gpu_name = training.get("gpu", {}).get("gpu_name", "CPU")
    train_sec = training.get("training_wall_sec", "(run Step 6 with updated script or note manually)")
    epochs = training.get("epochs_ran", training.get("best_epoch", "?"))

    md = f"""# Speed benchmark (simulation vs GNN surrogate)

## Training
| Item | Value |
|------|-------|
| GPU | {gpu_name} |
| Model | {training.get('model_type', 'gat')} |
| Epochs | {epochs} |
| Training wall time (s) | {train_sec} |

## Per-graph latency (n={n} sampled windows)
| Metric | Median | Mean |
|--------|--------|------|
| **Simulation** (Step 4, CPU) | **{sim_med:.3f} s** | {sim_mean:.3f} s |
| **ML inference** (GAT forward, {device}) | **{inf_med:.2f} ms** | {inf_mean:.2f} ms |
| **Speedup** | **{speedup_med:.0f}×** | {speedup_mean:.0f}× |

## Paper one-liner
Full attack-curve simulation: **{sim_med:.2f} s/graph** (CPU, NetworkX).
GNN surrogate inference: **{inf_med:.2f} ms/graph** ({gpu_name}).
**~{speedup_med:.0f}×** acceleration for label generation at inference time.

## Methods note
- Simulation: betweenness + capacity + {args.n_random}× random attack curves (10 removal points each), same as Step 4.
- Inference: single forward pass, 30-dim output (3×10-point curves); batch size 1.
"""
    md_path.write_text(md, encoding="utf-8")

    tracker = MlflowTracker(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or "step8_speed_benchmark",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"step": "8_benchmark"},
    )
    tracker.log_params(vars(args))
    tracker.log_metrics(
        {
            "simulation_sec_median": sim_med,
            "inference_ms_median": inf_med,
            "speedup_median_x": speedup_med,
            "simulation_sec_mean": sim_mean,
            "inference_ms_mean": inf_mean,
            "speedup_mean_x": speedup_mean,
        }
    )
    tracker.log_artifact(json_path)
    tracker.log_artifact(md_path)
    tracker.end()

    print("\n=== Summary (for paper) ===")
    print(f"Simulation:  {sim_med:.3f} s/graph (median)")
    print(f"Inference:   {inf_med:.2f} ms/graph (median)")
    print(f"Speedup:     {speedup_med:.0f}×")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
