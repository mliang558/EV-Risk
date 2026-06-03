#!/usr/bin/env python3
"""
Step 6: Train GAT (main) or GCN (baseline) on PyG window graphs.

Loss: MSE on 30-dim y (three 10-point attack curves).
Split: east (center_lon > threshold) train, west test.

Usage:
  python step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gat --device cuda
  python step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gcn --device cuda
  python step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model mlp --device cuda --out pyg/train_mlp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.edge_utils import sanitize_graph_list
from gnn.constants import parse_attacks
from gnn.metrics import curve_metrics, metrics_by_attack
from gnn.mlflow_utils import MlflowTracker
from gnn.models import ResilienceGNN
from gnn.train_utils import match_target_to_pred


def load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        elif tgt.dim() == 3 and tgt.size(1) == 1:
            tgt = tgt.squeeze(1)
        tgt = match_target_to_pred(tgt, pred, getattr(model, "attacks", parse_attacks("all")))
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(ps)
    k = getattr(model, "n_out_points", 10)
    attacks = getattr(model, "attacks", parse_attacks("all"))
    out = curve_metrics(y_true, y_pred, n_points=k)
    out.update(metrics_by_attack(y_true, y_pred, attacks=attacks, n_points=k))
    return out


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y, _, _, _ = model(batch)
        target = batch.y
        if target.dim() == 1:
            target = target.view(y.size(0), -1)
        elif target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)
        target = match_target_to_pred(target, y, getattr(model, "attacks", parse_attacks("all")))
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def main() -> None:
    print("Step 6: train GNN (use: python -u step6_train_gnn.py ...)", flush=True)
    parser = argparse.ArgumentParser(description="Step 6: train GAT/GCN")
    parser.add_argument("--dataset", default="data/step4_gpu/pyg/pyg_dataset.pt")
    parser.add_argument("--out", default=None, help="Checkpoint dir")
    parser.add_argument("--model", choices=["gat", "gcn", "mlp"], default="gat")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--curve-points",
        type=int,
        default=10,
        choices=[4, 10],
        help="Predict first k removal points (4 = 5%%–30%%; 10 = full curve)",
    )
    parser.add_argument("--split", default="east_west", choices=["east_west"])
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", default=True, help="Log to MLflow (default on)")
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", help="Disable MLflow")
    parser.add_argument("--mlflow-experiment", default="ev-charging-resilience-gnn")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None, help="e.g. file:./mlruns or http://...")
    parser.add_argument(
        "--global-features",
        action="store_true",
        help="Concat window-level graph stats (from subgraphs/*.npz) to GAT readout",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Dir with subgraphs/ (for --global-features if gf not in dataset)",
    )
    parser.add_argument(
        "--global-features-csv",
        default=None,
        help="Precomputed window_global_features.csv (recommended; avoids heavy NetworkX)",
    )
    parser.add_argument(
        "--global-fast",
        action="store_true",
        help="Lightweight npz features only (if no CSV); skips betweenness",
    )
    parser.add_argument(
        "--attack",
        choices=["betweenness", "capacity", "random"],
        default=None,
        help="Single-task: one attack head only (10-dim y). Overrides --attacks.",
    )
    parser.add_argument(
        "--attacks",
        default="all",
        help="Multi-head: all | no-capacity | betweenness | random | betweenness,random",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Skip edge/num_nodes sanitize (use if plain GAT already trained on same .pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, attach global features, run one train batch, then exit (debug Bus error)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N graphs (debug memory; 0 = all)",
    )
    args = parser.parse_args()
    if args.attack:
        attack_tuple = (args.attack,)
    else:
        attack_tuple = parse_attacks(args.attacks)
    n_attacks = len(attack_tuple)
    single_task = n_attacks == 1
    use_mlflow = bool(getattr(args, "mlflow", True))

    import os

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"[Step6] pid={os.getpid()} device={device} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    try:
        import shutil

        _free = shutil.disk_usage(Path.cwd()).free
        print(f"[Step6] disk_free_gb={_free / (1024**3):.2f}", flush=True)
        if _free < 500 * 1024 * 1024:
            print(
                "[Step6] WARN: disk almost full — torch.load/save may Bus-error; "
                "free space or use --no-mlflow --out /tmp/train_gat_global",
                flush=True,
            )
    except Exception:
        pass

    ds_path = Path(args.dataset).resolve()
    print(f"[Step6] Loading {ds_path} (may take 1–3 min on 10k graphs)...", flush=True)
    t_load = time.perf_counter()
    bundle = load_bundle(ds_path)
    graphs = bundle["graphs"]
    print(f"[Step6] Loaded {len(graphs)} graphs in {time.perf_counter() - t_load:.1f}s", flush=True)

    if args.limit and args.limit > 0:
        graphs = graphs[: args.limit]
        print(f"[Step6] --limit {args.limit}: using subset only", flush=True)

    split_masks = bundle.get("split_masks", {})
    train_mask = split_masks["east_west_train"]
    test_mask = split_masks["east_west_test"]
    if args.limit and args.limit > 0:
        train_mask = train_mask[: args.limit]
        test_mask = test_mask[: args.limit]

    global_dim = 0
    if args.global_features:
        if not hasattr(graphs[0], "gf"):
            data_dir = Path(args.data_dir).resolve()
            if str(args.data_dir) in (".", "./"):
                data_dir = Path.cwd().resolve()
            csv_path = Path(args.global_features_csv) if args.global_features_csv else None
            if csv_path is None:
                for cand in (
                    data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
                    data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
                    data_dir / "window_global_features.csv",
                ):
                    if cand.is_file():
                        csv_path = cand
                        break
            if csv_path is not None and csv_path.is_file():
                from gnn.global_graph_features import attach_global_features_from_csv

                gf_stats = attach_global_features_from_csv(graphs, csv_path, train_mask)
            else:
                from gnn.global_graph_features import attach_global_features_to_graphs

                if not args.global_fast:
                    print(
                        "[Step6] WARN: full NetworkX features on 10k windows may OOM/bus-error; "
                        "use --global-features-csv or --global-fast",
                        flush=True,
                    )
                gf_stats = attach_global_features_to_graphs(
                    graphs, data_dir, train_mask, fast=args.global_fast
                )
            bundle.setdefault("stats", {}).update(gf_stats)
        global_dim = int(bundle.get("stats", {}).get("global_feat_dim", 0))
        if global_dim <= 0 and hasattr(graphs[0], "gf"):
            global_dim = int(graphs[0].gf.numel())
        print(f"[Step6] Global features dim={global_dim}", flush=True)

    if not args.no_sanitize:
        print("[Step6] Sanitizing edges / num_nodes for GAT...", flush=True)
        t_san = time.perf_counter()
        n_fixed = sanitize_graph_list(graphs)
        print(f"[Step6] Sanitize done in {time.perf_counter() - t_san:.1f}s", flush=True)
        if n_fixed:
            print(
                f"[Step6] Sanitized edges/num_nodes on {n_fixed} graphs "
                f"(fixes GAT batch / global_mean_pool).",
                flush=True,
            )
    else:
        print("[Step6] Skipping sanitize (--no-sanitize)", flush=True)

    train_graphs = [graphs[i] for i in range(len(graphs)) if train_mask[i]]
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
    print(
        f"Train {len(train_graphs)} | Test {len(test_graphs)} | Model {args.model} | "
        f"attacks={','.join(attack_tuple)} ({n_attacks} heads)",
        flush=True,
    )

    edge_dim = 1 if args.model == "gat" else 0
    train_bs = args.batch_size if args.model != "gat" else min(args.batch_size, 32)
    test_bs = args.batch_size if args.model != "gat" else 1
    print(f"[Step6] Building DataLoaders (train_bs={train_bs}, test_bs={test_bs})...", flush=True)

    train_loader = DataLoader(train_graphs, batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=test_bs, shuffle=False)

    if args.dry_run:
        print("[Step6] --dry-run: one forward + backward on train batch, then exit", flush=True)
        model = ResilienceGNN(
            in_channels=4,
            hidden_channels=args.hidden,
            gat_heads=args.gat_heads,
            edge_dim=edge_dim,
            model_type=args.model,
            n_out_points=args.curve_points,
            global_dim=global_dim,
            attacks=attack_tuple,
        ).to(device)
        batch = next(iter(train_loader)).to(device)
        print(
            f"[Step6] batch x={batch.x.shape} ei={batch.edge_index.shape} "
            f"gf={getattr(batch, 'gf', None)}",
            flush=True,
        )
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        y, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(y.size(0), -1)
        elif tgt.dim() == 3 and tgt.size(1) == 1:
            tgt = tgt.squeeze(1)
        tgt = match_target_to_pred(tgt, y, attack_tuple)
        loss = crit(y, tgt)
        loss.backward()
        opt.step()
        print(f"[Step6] dry-run OK loss={float(loss.item()):.6f}", flush=True)
        return

    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=args.hidden,
        gat_heads=args.gat_heads,
        edge_dim=edge_dim,
        model_type=args.model,
        n_out_points=args.curve_points,
        global_dim=global_dim,
        attacks=attack_tuple,
    ).to(device)
    print(
        f"[Step6] Output: {args.curve_points} pts/head × {n_attacks} attacks "
        f"({n_attacks * args.curve_points}-dim y)"
        + (f" | GAT+global(F={global_dim})" if global_dim else ""),
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if args.out:
        out_dir = Path(args.out)
    else:
        suffix = f"_{args.model}" if args.curve_points == 10 else f"_{args.model}_p{args.curve_points}"
        if args.global_features and args.model == "gat":
            suffix = f"{suffix}_global"
        if attack_tuple != parse_attacks("all"):
            suffix = f"{suffix}_{attack_tuple[0]}" if single_task else f"{suffix}_{'-'.join(attack_tuple)}"
        out_dir = Path(args.dataset).parent / f"train{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = MlflowTracker(
        enabled=use_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or f"step6_{args.model}",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"step": "6_train", "model": args.model},
    )
    tracker.log_params(
        {
            **vars(args),
            "n_train": len(train_graphs),
            "n_test": len(test_graphs),
            "dataset": str(Path(args.dataset).resolve()),
            "out_dir": str(out_dir.resolve()),
        }
    )

    best_mae = float("inf")
    stale = 0
    history = []
    train_t0 = time.perf_counter()
    gpu_meta = {}
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_meta = {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
        }

    print(f"[Step6] Starting training on {device} (epoch progress below)...", flush=True)
    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc=f"Step6 {args.model}", unit="epoch", file=sys.stderr)

    for epoch in epoch_iter:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(mae=f"{metrics['mae']:.4f}", r2=f"{metrics['r2']:.3f}")
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "test_r2": metrics["r2"],
            "test_mae": metrics["mae"],
        }
        for head in attack_tuple:
            row[f"test_{head}_r2"] = metrics[head]["r2"]
            row[f"test_{head}_mae"] = metrics[head]["mae"]
        history.append(row)
        log_row = {
            "train_mse": train_loss,
            "test_mae": metrics["mae"],
            "test_r2": metrics["r2"],
        }
        for head in attack_tuple:
            log_row[f"test_{head}_r2"] = metrics[head]["r2"]
            log_row[f"test_{head}_mae"] = metrics[head]["mae"]
        tracker.log_metrics(log_row, step=epoch)

        head_r2 = " ".join(f"{h[0].upper()}={metrics[h]['r2']:.3f}" for h in attack_tuple)
        print(
            f"Epoch {epoch:3d} | train_mse={train_loss:.5f} | "
            f"test_mae={metrics['mae']:.4f} r2={metrics['r2']:.4f} | {head_r2}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            stale = 0
            ckpt = {
                "model": model.state_dict(),
                "model_type": args.model,
                "hidden": args.hidden,
                "gat_heads": args.gat_heads,
                "curve_points": args.curve_points,
                "global_dim": global_dim,
                "attacks": list(attack_tuple),
                "epoch": epoch,
                "test_metrics": metrics,
                "stats": bundle.get("stats", {}),
            }
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    train_wall_sec = time.perf_counter() - train_t0
    best_epoch = history[-1]["epoch"] if history else 0
    ckpt_path = out_dir / "best_model.pt"
    if ckpt_path.exists():
        try:
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(ckpt_path, map_location="cpu")
        best_epoch = int(saved.get("epoch", best_epoch))

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    train_meta = {
        "model_type": args.model,
        "curve_points": args.curve_points,
        "epochs_ran": len(history),
        "epochs_max": args.epochs,
        "best_epoch": best_epoch,
        "training_wall_sec": round(train_wall_sec, 2),
        "best_test_mae": best_mae,
        **gpu_meta,
    }
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    tracker.log_metrics(
        {
            "best_test_mae": best_mae,
            "training_wall_sec": train_wall_sec,
            "best_epoch": float(best_epoch),
            "epochs_ran": float(len(history)),
        }
    )
    tracker.log_artifact(out_dir / "best_model.pt")
    tracker.log_artifact(out_dir / "history.json")
    tracker.log_artifact(out_dir / "train_meta.json")
    if use_mlflow and ckpt_path.exists():
        tracker.log_model_pytorch(model.cpu())
    tracker.end()

    print(f"Best test MAE={best_mae:.4f} | checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Training time: {train_wall_sec:.1f}s ({len(history)} epochs) | GPU: {gpu_meta.get('gpu_name', 'cpu')}")
    if tracker.run_id:
        print(f"MLflow run_id: {tracker.run_id}")


if __name__ == "__main__":
    main()
