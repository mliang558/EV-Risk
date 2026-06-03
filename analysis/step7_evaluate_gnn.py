#!/usr/bin/env python3
"""
Step 7: Evaluate trained GNN.

- East/West test metrics (R², MAE per head)
- Leave-one-state-out generalization
- GAT edge attention visualization (top edges per window)

Usage:
  python step7_evaluate_gnn.py \\
    --dataset pyg/pyg_dataset.pt \\
    --checkpoint train_gat/best_model.pt \\
    --out results_gnn/gat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges, sanitize_graph_list
from gnn.constants import ATTACK_ORDER, parse_attacks
from gnn.metrics import curve_metrics, metrics_by_attack
from gnn.mlflow_utils import MlflowTracker
from gnn.models import ResilienceGNN
from gnn.train_utils import match_target_to_pred
from gnn.splits import leave_one_state_folds, primary_state


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(ckpt_path: Path, device: torch.device) -> ResilienceGNN:
    ckpt = _torch_load(ckpt_path, device)
    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=int(ckpt.get("hidden", 32)),
        gat_heads=int(ckpt.get("gat_heads", 4)),
        edge_dim=1 if ckpt.get("model_type", "gat") == "gat" else 0,
        model_type=ckpt.get("model_type", "gat"),
        n_out_points=int(ckpt.get("curve_points", 10)),
        global_dim=int(ckpt.get("global_dim", 0)),
        attacks=tuple(ckpt.get("attacks", ATTACK_ORDER)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, graphs, device, batch_size: int = 64):
    from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges

    for g in graphs:
        sanitize_graph_edges(g)
        ensure_graph_level_y(g)

    # GAT + batched graphs: use small batches (edge_attr must match per graph)
    bs = 1 if getattr(model, "model_type", "") == "gat" else batch_size
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ys, ps, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        elif tgt.dim() == 3 and tgt.size(1) == 1:
            tgt = tgt.squeeze(1)
        tgt = match_target_to_pred(tgt, pred, getattr(model, "attacks", ATTACK_ORDER))
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(list(batch.window_id))
    return np.vstack(ys), np.vstack(ps), ids


def plot_pred_vs_true(y_true, y_pred, out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Step7] Skip plot (pip install matplotlib for PNG)", flush=True)
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true.ravel(), y_pred.ravel(), alpha=0.15, s=8)
    lim = max(y_true.max(), y_pred.max(), 1.0)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    m = curve_metrics(y_true, y_pred)
    ax.set_xlabel("True relative loss")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nR²={m['r2']:.3f} MAE={m['mae']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_attention_map(model, data, meta_row, out_path: Path, top_k: int = 30):
    if model.model_type != "gat" or model._last_attn is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    attn = model._last_attn
    edge_index = attn["edge_index"].numpy()
    alpha = attn["alpha"].numpy().mean(axis=-1).flatten()

    order = np.argsort(-alpha)[:top_k]
    top_ei = edge_index[:, order]
    top_a = alpha[order]

    x = data.x.numpy()
    lon = x[:, 0]
    lat = x[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    # Node colors: capacity (col 2 of x); y_betweenness is graph-level [1,10], not per-node
    node_color = x[:, 2] if x.shape[1] > 2 else np.zeros(len(lon))
    sc = ax.scatter(lon, lat, c=node_color, cmap="viridis", s=40, zorder=2)
    plt.colorbar(sc, ax=ax, label="capacity (norm)", fraction=0.046, pad=0.04)
    for j in range(top_ei.shape[1]):
        u, v = int(top_ei[0, j]), int(top_ei[1, j])
        ax.plot(
            [lon[u], lon[v]],
            [lat[u], lat[v]],
            color="red",
            alpha=float(min(1.0, 0.3 + top_a[j] * 5)),
            linewidth=1 + 2 * top_a[j],
            zorder=1,
        )
    ax.set_xlabel("lon (norm)")
    ax.set_ylabel("lat (norm)")
    ax.set_title(f"Top-{top_k} GAT edges | {meta_row.get('window_id', '')}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def leave_one_state_out(
    model_cls_kwargs: dict,
    graphs: list,
    meta_df: pd.DataFrame,
    device: torch.device,
    min_windows: int,
    epochs: int,
    batch_size: int,
) -> pd.DataFrame:
    manifest = meta_df.copy()
    manifest["primary_state"] = manifest.apply(primary_state, axis=1)
    folds = leave_one_state_folds(manifest, min_windows=min_windows)
    rows = []

    fold_items = list(folds.items())
    if tqdm is not None:
        fold_items = tqdm(fold_items, desc="LOO by state", unit="state")

    for st, test_mask in fold_items:
        train_graphs = [graphs[i] for i in range(len(graphs)) if not test_mask[i]]
        test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
        if len(test_graphs) < 5:
            continue

        model = ResilienceGNN(**model_cls_kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad()
                pred, _, _, _ = model(batch)
                tgt = batch.y
                if tgt.dim() == 1:
                    tgt = tgt.view(pred.size(0), -1)
                crit(pred, tgt).backward()
                opt.step()

        model.eval()
        y_true, y_pred, _ = predict_all(model, test_graphs, device, batch_size)
        m = curve_metrics(y_true, y_pred)
        rows.append({"held_out_state": st, "n_test": len(test_graphs), **m})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: evaluate GNN")
    parser.add_argument("--dataset", default="data/step4_gpu/pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=".", help="subgraphs/ for GAT+global eval")
    parser.add_argument(
        "--global-features-csv",
        default=None,
        help="window_global_features.csv (same as Step 6; preferred over npz rebuild)",
    )
    parser.add_argument("--out", default="results_gnn/eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-attn-plots", type=int, default=12, help="GAT attention maps to save")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG figures (metrics JSON only)")
    parser.add_argument("--loo", action="store_true", help="Run leave-one-state-out (slow)")
    parser.add_argument("--loo-epochs", type=int, default=40)
    parser.add_argument("--loo-min-windows", type=int, default=20)
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", default=True)
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    parser.add_argument("--mlflow-experiment", default="ev-charging-resilience-gnn")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Step7] Loading dataset...", flush=True)
    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    print("[Step7] Sanitizing graphs (same as Step 6)...", flush=True)
    sanitize_graph_list(graphs)
    meta_df = bundle["meta"]
    split_masks = bundle.get("split_masks", {})

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = _torch_load(ckpt_path, "cpu")
    model = load_model(ckpt_path, device)
    print(f"[Step7] Model={model.model_type} checkpoint={ckpt_path}", flush=True)

    if int(getattr(model, "global_dim", 0)) > 0 and not hasattr(graphs[0], "gf"):
        data_dir = Path(args.data_dir).resolve()
        if str(args.data_dir) in (".", "./"):
            data_dir = Path.cwd().resolve()
        train_mask = split_masks.get("east_west_train", np.ones(len(graphs), bool))
        csv_path = Path(args.global_features_csv) if args.global_features_csv else None
        if csv_path is None:
            for cand in (
                data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
                data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
            ):
                if cand.is_file():
                    csv_path = cand
                    break
        if csv_path is not None and csv_path.is_file():
            from gnn.global_graph_features import attach_global_features_from_csv

            attach_global_features_from_csv(graphs, csv_path, train_mask, verbose=True)
        else:
            from gnn.global_graph_features import attach_global_features_to_graphs

            attach_global_features_to_graphs(
                graphs, data_dir, train_mask, fast=True, verbose=True
            )

    test_idx = [i for i in range(len(graphs)) if split_masks.get("east_west_test", np.zeros(len(graphs), bool))[i]]
    test_graphs = [graphs[i] for i in test_idx]
    y_true, y_pred, ids = predict_all(model, test_graphs, device, args.batch_size)

    attacks = getattr(model, "attacks", ATTACK_ORDER)
    k = getattr(model, "n_out_points", 10)
    metrics = curve_metrics(y_true, y_pred, n_points=k)
    report = {
        "split": "east_west_test",
        "n_test": len(test_graphs),
        "attacks": list(attacks),
        "overall": metrics,
        **metrics_by_attack(y_true, y_pred, attacks=attacks, n_points=k),
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    tracker = MlflowTracker(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or "step7_eval",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"step": "7_eval", "checkpoint": str(ckpt_path)},
    )
    tracker.log_params({"checkpoint": str(ckpt_path), "dataset": str(args.dataset), "n_test": len(test_graphs)})
    ml = {"test_r2": metrics["r2"], "test_mae": metrics["mae"]}
    for head in attacks:
        if head in report:
            ml[f"test_{head}_r2"] = report[head]["r2"]
    tracker.log_metrics(ml)
    tracker.log_artifact(out_dir / "test_metrics.json")

    print("=== West test set ===")
    print(f"  Overall R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f}")
    for head in attacks:
        if head in report:
            h = report[head]
            print(f"  {head}: R²={h['r2']:.4f} MAE={h['mae']:.4f}")

    dim = y_true.shape[1]
    png_path = out_dir / "pred_vs_true_west.png"
    if not args.no_plots:
        plot_pred_vs_true(y_true, y_pred, png_path, f"West test ({dim}-dim)")
    if png_path.is_file():
        tracker.log_artifact(png_path)
    tracker.end()
    print(f"\nSaved -> {out_dir / 'test_metrics.json'}", flush=True)

    pred_df = meta_df[meta_df["window_id"].isin(ids)].copy()
    pred_df["y_true_mean"] = y_true.mean(axis=1)
    pred_df["y_pred_mean"] = y_pred.mean(axis=1)
    pred_df.to_csv(out_dir / "west_predictions.csv", index=False)

    if model.model_type == "gat" and args.n_attn_plots > 0:
        attn_dir = out_dir / "attention_maps"
        attn_dir.mkdir(exist_ok=True)
        meta_by_id = {r["window_id"]: r for _, r in meta_df.iterrows()}
        for g in test_graphs[: args.n_attn_plots]:
            try:
                sanitize_graph_edges(g)
                ensure_graph_level_y(g)
                batch = Batch.from_data_list([g]).to(device)
                model(batch)
                row = meta_by_id.get(g.window_id, {"window_id": g.window_id})
                plot_attention_map(
                    model,
                    g,
                    row,
                    attn_dir / f"{g.window_id}_attn.png",
                )
            except Exception as e:
                print(f"[Step7] Skip attention map {g.window_id}: {e}", flush=True)
        print(f"Saved attention maps -> {attn_dir}")

    if args.loo:
        print("=== Leave-one-state-out (retrain per fold) ===")
        model_kw = {
            "in_channels": 4,
            "hidden_channels": int(ckpt.get("hidden", 32)),
            "gat_heads": int(ckpt.get("gat_heads", 4)),
            "edge_dim": 1 if ckpt.get("model_type") == "gat" else 0,
            "model_type": ckpt.get("model_type", "gat"),
        }
        loo_df = leave_one_state_out(
            model_kw,
            graphs,
            meta_df,
            device,
            args.loo_min_windows,
            args.loo_epochs,
            args.batch_size,
        )
        loo_df.to_csv(out_dir / "leave_one_state_out.csv", index=False)
        print(loo_df.describe())
        print(f"LOO summary -> {out_dir / 'leave_one_state_out.csv'}")

    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
