#!/usr/bin/env python3
"""
Geographic generalization: region holdout, east-west retrain, or partial/full LOO.

Fast paths (recommended):
  # One split: train West+Midwest+South, test Northeast (~hours)
  --mode region --holdout-region northeast

  # Main paper east/west (retrain once)
  --mode east_west

  # Appendix: 5 representative states only
  --mode loo --states DE,CA,TX,ME,IA

Full 48-state LOO (slow):
  --mode loo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import parse_attacks
from gnn.edge_utils import sanitize_graph_list
from gnn.metrics import curve_metrics, metrics_by_attack
from gnn.models import ResilienceGNN, model_uses_edge_attr, resilience_kwargs_from_ckpt
from gnn.splits import (
    REPRESENTATIVE_LOO_STATES,
    east_west_retrain_masks,
    holdout_region_masks,
    leave_one_state_folds,
    primary_state,
)
from gnn.train_utils import match_target_to_pred

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(requested: str | None, data_dir: Path) -> Path:
    """Architecture template only (weights retrained per fold). Auto-pick if missing."""
    if requested:
        p = Path(requested)
        if not p.is_absolute():
            p = (data_dir / p).resolve()
        if p.is_file():
            return p
    candidates = [
        "pyg/train_gat_global_L5_ext_br/best_model.pt",
        "pyg/train_gat_global_L5_ext/best_model.pt",
        "pyg/train_gat_global_br/best_model.pt",
        "pyg/train_gat_global/best_model.pt",
        "pyg/train_gat_global_betweenness/best_model.pt",
    ]
    for rel in candidates:
        p = (data_dir / rel).resolve()
        if p.is_file():
            print(f"[geo] checkpoint not found at {requested!r}; using {p}", flush=True)
            return p
    tried = ", ".join(str(data_dir / c) for c in candidates)
    raise FileNotFoundError(
        f"No checkpoint for architecture template. Train step6 first, e.g.\n"
        f"  pyg/train_gat_global_L5_ext_br/best_model.pt\n"
        f"Searched: {tried}"
    )


def attach_global_if_needed(
    graphs,
    data_dir: Path,
    train_mask: np.ndarray,
    csv_path: Path | None,
    *,
    global_dim: int,
    ckpt: dict | None,
) -> None:
    if global_dim <= 0:
        return
    from gnn.global_graph_features import ensure_global_features_for_eval

    ensure_global_features_for_eval(
        graphs,
        global_dim,
        train_mask,
        data_dir,
        ckpt=ckpt,
        csv_path=csv_path,
        force_reattach=True,
        verbose=False,
    )


@torch.no_grad()
def predict_graphs(model, graphs, device, attacks, batch_size: int, desc: str = "Predict"):
    bs = 1 if model_uses_edge_attr(model.model_type) else batch_size
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ys, ps = [], []
    batch_it = loader
    if tqdm is not None:
        batch_it = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in batch_it:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        tgt = match_target_to_pred(tgt, pred, attacks)
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return np.vstack(ys), np.vstack(ps)


def train_fold(
    model_kw: dict,
    train_graphs: list,
    device: torch.device,
    attacks: tuple[str, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    split_label: str = "train",
) -> ResilienceGNN:
    model = ResilienceGNN(**model_kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    bs = min(batch_size, 32) if model_uses_edge_attr(model_kw.get("model_type", "gat")) else batch_size
    loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)

    epoch_it = range(1, epochs + 1)
    if tqdm is not None:
        epoch_it = tqdm(epoch_it, desc=f"Train {split_label}", unit="epoch")

    for epoch in epoch_it:
        model.train()
        loss_sum, n_graphs = 0.0, 0
        batch_it = loader
        if tqdm is not None and epochs <= 5:
            batch_it = tqdm(loader, desc=f"  ep{epoch} batches", unit="batch", leave=False)
        for batch in batch_it:
            batch = batch.to(device)
            opt.zero_grad()
            pred, _, _, _ = model(batch)
            tgt = batch.y
            if tgt.dim() == 1:
                tgt = tgt.view(pred.size(0), -1)
            tgt = match_target_to_pred(tgt, pred, attacks)
            loss = crit(pred, tgt)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * batch.num_graphs
            n_graphs += batch.num_graphs
        if tqdm is not None and hasattr(epoch_it, "set_postfix"):
            epoch_it.set_postfix(mse=f"{loss_sum / max(n_graphs, 1):.4f}")
    return model


def run_single_split(
    split_name: str,
    train_idx: list[int],
    test_idx: list[int],
    graphs: list,
    model_kw: dict,
    device: torch.device,
    attacks: tuple[str, ...],
    k: int,
    args,
    data_dir: Path,
    gcsv: Path | None,
) -> dict:
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    tr_mask = np.zeros(len(graphs), dtype=bool)
    tr_mask[train_idx] = True
    attach_global_if_needed(
        graphs,
        data_dir,
        tr_mask,
        gcsv,
        global_dim=int(model_kw.get("global_dim", 0)),
        ckpt=ckpt,
    )
    print(f"[geo] >> {split_name} | train={len(train_graphs)} test={len(test_graphs)}", flush=True)
    model = train_fold(
        model_kw,
        train_graphs,
        device,
        attacks,
        args.epochs,
        args.batch_size,
        args.lr,
        split_label=split_name,
    )
    model.eval()
    y_true, y_pred = predict_graphs(
        model, test_graphs, device, attacks, args.batch_size, desc=f"Eval {split_name}"
    )
    overall = curve_metrics(y_true, y_pred, n_points=k)
    by_head = metrics_by_attack(y_true, y_pred, attacks=attacks, n_points=k)
    row = {
        "split": split_name,
        "n_test": len(test_graphs),
        "n_train": len(train_graphs),
        "overall_r2": overall["r2"],
        "overall_mae": overall["mae"],
    }
    for a in attacks:
        row[f"{a}_r2"] = by_head[a]["r2"]
        row[f"{a}_mae"] = by_head[a]["mae"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Geographic generalization (region / LOO / east-west)")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Architecture template (.pt); auto-searches pyg/train_gat_* if missing",
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--global-features-csv", default=None)
    parser.add_argument("--out", default="results_gnn/loo")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-states", type=int, default=0, help="Debug: only first N LOO states")
    parser.add_argument(
        "--mode",
        choices=["loo", "region", "east_west"],
        default="loo",
        help="loo=state folds | region=hold out one census region | east_west=lon split",
    )
    parser.add_argument(
        "--holdout-region",
        choices=["northeast", "midwest", "south", "west"],
        default="northeast",
        help="For --mode region: train on all other regions, test this one",
    )
    parser.add_argument(
        "--states",
        default=None,
        help="For --mode loo: comma-separated state codes only (e.g. DE,CA,TX,ME,IA)",
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help="Shorthand: --mode loo --states DE,CA,TX,ME,IA",
    )
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument(
        "--attacks",
        default=None,
        help="Override ckpt attacks (default: from checkpoint or no-capacity)",
    )
    args = parser.parse_args()
    if args.representative:
        args.mode = "loo"
        args.states = ",".join(REPRESENTATIVE_LOO_STATES)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    ckpt_path = resolve_checkpoint_path(args.checkpoint, data_dir)
    ckpt = _torch_load(ckpt_path, "cpu")
    model_kw = resilience_kwargs_from_ckpt(ckpt)
    print(f"[geo] arch from {ckpt_path} | {model_kw.get('model_type')} L={model_kw.get('num_layers')}", flush=True)
    attacks = parse_attacks(args.attacks) if args.attacks else tuple(ckpt.get("attacks", parse_attacks("no-capacity")))
    k = int(model_kw.get("n_out_points", 10))

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs, show_progress=False)
    meta_df = bundle["meta"].copy()
    meta_df["primary_state"] = meta_df.apply(primary_state, axis=1)

    gcsv = Path(args.global_features_csv) if args.global_features_csv else None
    if gcsv is None:
        for cand in (
            data_dir / "results_gnn/global_extended/window_global_features.csv",
            data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
        ):
            if cand.is_file():
                gcsv = cand
                break

    rows: list[dict] = []

    if args.mode == "region":
        train_mask, test_mask, label = holdout_region_masks(meta_df, args.holdout_region)
        train_idx = [i for i in range(len(graphs)) if train_mask[i]]
        test_idx = [i for i in range(len(graphs)) if test_mask[i]]
        print(
            f"[geo] region holdout={args.holdout_region} train={len(train_idx)} test={len(test_idx)}",
            flush=True,
        )
        rows.append(
            run_single_split(
                label, train_idx, test_idx, graphs, model_kw, device, attacks, k, args, data_dir, gcsv
            )
        )
    elif args.mode == "east_west":
        train_mask, test_mask, label = east_west_retrain_masks(meta_df, args.lon_threshold)
        train_idx = [i for i in range(len(graphs)) if train_mask[i]]
        test_idx = [i for i in range(len(graphs)) if test_mask[i]]
        print(
            f"[geo] east_west train={len(train_idx)} test={len(test_idx)} (lon>{args.lon_threshold})",
            flush=True,
        )
        rows.append(
            run_single_split(
                label,
                train_idx,
                test_idx,
                graphs,
                model_kw,
                device,
                attacks,
                k,
                args,
                data_dir,
                gcsv,
                ckpt,
            )
        )
    else:
        states_only = None
        if args.states:
            states_only = tuple(s.strip() for s in args.states.split(",") if s.strip())
        folds = leave_one_state_folds(meta_df, min_windows=args.min_windows, states_only=states_only)
        fold_items = sorted(folds.items(), key=lambda x: x[0])
        if args.limit_states and args.limit_states > 0:
            fold_items = fold_items[: args.limit_states]
        print(
            f"[LOO] states={len(fold_items)} epochs={args.epochs} attacks={attacks}",
            flush=True,
        )
        it = fold_items
        if tqdm is not None:
            it = tqdm(fold_items, desc="LOO states", unit="state")
        for st, test_mask in it:
            train_idx = [i for i in range(len(graphs)) if not test_mask[i]]
            test_idx = [i for i in range(len(graphs)) if test_mask[i]]
            row = run_single_split(
                f"loo_{st}",
                train_idx,
                test_idx,
                graphs,
                model_kw,
                device,
                attacks,
                k,
                args,
                data_dir,
                gcsv,
                ckpt,
            )
            row["held_out_state"] = st
            rows.append(row)
            print(
                f"  {st}: n_test={row['n_test']} overall_R2={row['overall_r2']:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "loo_by_state.csv", index=False)

    summary = {
        "checkpoint_arch": str(ckpt_path),
        "attacks": list(attacks),
        "epochs": args.epochs,
        "n_states": len(df),
        "mean_overall_r2": float(df["overall_r2"].mean()) if len(df) else np.nan,
        "std_overall_r2": float(df["overall_r2"].std()) if len(df) > 1 else 0.0,
    }
    for a in attacks:
        col = f"{a}_r2"
        if col in df.columns:
            summary[f"mean_{a}_r2"] = float(df[col].mean())

    with open(out_dir / "loo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "| Split | n_test | Betweenness R² | Random R² | Overall R² |",
        "|-------|--------|----------------|-----------|------------|",
    ]
    sort_col = "held_out_state" if "held_out_state" in df.columns else "split"
    for _, r in df.sort_values("overall_r2", ascending=False).iterrows():
        name = r.get("held_out_state", r.get("split", ""))
        b = r.get("betweenness_r2", np.nan)
        rnd = r.get("random_r2", np.nan)
        md_lines.append(
            f"| {name} | {int(r['n_test'])} | {b:.3f} | {rnd:.3f} | {r['overall_r2']:.3f} |"
        )
    md_lines.append(
        f"\n**Mean overall R²** = {summary['mean_overall_r2']:.3f} "
        f"(± {summary['std_overall_r2']:.3f}) across {summary['n_states']} states"
    )
    (out_dir / "loo_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n[LOO] mean overall R² = {summary['mean_overall_r2']:.4f}")
    print(f"Saved -> {out_dir}/loo_by_state.csv , loo_table.md")


if __name__ == "__main__":
    main()
