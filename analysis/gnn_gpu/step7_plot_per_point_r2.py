#!/usr/bin/env python3
"""
Plot R² / MAE at each removal point (5%–90%) on West test.
Self-contained: no gnn.train_utils / no new constants required.

Usage (run from step4_gpu root, NOT inside gnn/):
  cd /opt/data_repo/mliang_work/step4_gpu
  python step7_plot_per_point_r2.py \\
    --dataset pyg/pyg_dataset.pt \\
    --checkpoint pyg/train_gat/best_model.pt \\
    --out results_gnn/per_point_r2 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

# --- defaults (no import from gnn.constants required) ---
N_CURVE = 10
PCT_REMOVAL = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
EARLY_K = 4
SCRIPT_VERSION = "standalone-v3"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def match_target_to_pred(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if target.dim() == 1:
        target = target.view(pred.size(0), -1)
    k = pred.size(1) // 3
    if target.size(1) == pred.size(1):
        return target
    if target.size(1) == 30 and pred.size(1) == 3 * k:
        return target.view(target.size(0), 3, N_CURVE)[:, :, :k].reshape(target.size(0), 3 * k)
    return target[:, : pred.size(1)]


def per_point_table(y_true_30: np.ndarray, y_pred: np.ndarray, k_model: int) -> pd.DataFrame:
    rows = []
    for a_idx, attack in enumerate(("betweenness", "capacity", "random")):
        for i in range(N_CURVE):
            yt = y_true_30[:, a_idx * N_CURVE + i]
            row = {
                "attack": attack,
                "point_index": i,
                "removal_pct": PCT_REMOVAL[i],
                "model_predicts": i < k_model,
            }
            if i < k_model:
                yp = y_pred[:, a_idx * k_model + i]
                row["r2"] = float(r2_score(yt, yp))
                row["mae"] = float(mean_absolute_error(yt, yp))
            else:
                row["r2"] = np.nan
                row["mae"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_per_point_r2(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"betweenness": "#e74c3c", "capacity": "#3498db", "random": "#2ecc71"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("r2", "mae")):
        for attack, sub in df.groupby("attack"):
            sub = sub.sort_values("point_index")
            m = sub["model_predicts"] if "model_predicts" in sub.columns else sub[metric].notna()
            ax.plot(
                sub.loc[m, "removal_pct"],
                sub.loc[m, metric],
                "o-",
                label=attack,
                color=colors.get(attack, "gray"),
                lw=2,
            )
        valid = df[df["model_predicts"]] if "model_predicts" in df.columns else df[df["r2"].notna()]
        if len(valid):
            mean_df = valid.groupby("removal_pct", as_index=False)[metric].mean()
            ax.plot(mean_df["removal_pct"], mean_df[metric], "k--", marker="s", label="mean", lw=1.5)
        ax.axvline(PCT_REMOVAL[EARLY_K - 1], color="gray", ls=":", lw=1.5)
        ax.set_xlabel("Node removal (%)")
        ax.set_ylabel("R²" if metric == "r2" else "MAE")
        ax.set_title(f"{metric.upper()} by removal point (West)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric == "r2":
            ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f"First {EARLY_K} points = 5%–{PCT_REMOVAL[EARLY_K-1]}%  |  {SCRIPT_VERSION}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_model(ckpt_path: Path, device: torch.device):
    from gnn.models import ResilienceGNN

    ckpt = _torch_load(ckpt_path, device)
    k = int(ckpt.get("curve_points", 10))
    kw = dict(
        in_channels=4,
        hidden_channels=int(ckpt.get("hidden", 32)),
        gat_heads=int(ckpt.get("gat_heads", 4)),
        edge_dim=1 if ckpt.get("model_type", "gat") == "gat" else 0,
        model_type=ckpt.get("model_type", "gat"),
    )
    try:
        model = ResilienceGNN(**kw, n_out_points=k)
    except TypeError:
        model = ResilienceGNN(**kw)
        k = 10
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if not hasattr(model, "n_out_points"):
        model.n_out_points = k
    return model, k


@torch.no_grad()
def predict_west(model, test_graphs, device) -> np.ndarray:
    try:
        from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges, sanitize_graph_list

        sanitize_graph_list(test_graphs, show_progress=False)
        for g in test_graphs:
            if getattr(model, "model_type", "") == "gat":
                sanitize_graph_edges(g)
            ensure_graph_level_y(g)
    except Exception as e:
        print(f"[warn] edge sanitize skipped: {e}", flush=True)

    bs = 1 if getattr(model, "model_type", "") == "gat" else 64
    loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)
    ps = []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        ps.append(pred.cpu().numpy())
    return np.vstack(ps)


def main() -> None:
    parser = argparse.ArgumentParser(description="R² per removal point (standalone)")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results_gnn/per_point_r2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[step7_plot] {SCRIPT_VERSION}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    test_mask = bundle.get("split_masks", {}).get("east_west_test", np.zeros(len(graphs), bool))
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
    print(f"West test graphs: {len(test_graphs)}", flush=True)

    model, k_model = load_model(Path(args.checkpoint).resolve(), device)
    y_true_30 = np.vstack([g.y.view(-1).numpy()[:30] for g in test_graphs])
    y_pred = predict_west(model, test_graphs, device)

    ppt = per_point_table(y_true_30, y_pred, k_model)
    ppt.to_csv(out_dir / "r2_mae_per_point.csv", index=False)
    plot_per_point_r2(ppt, out_dir / "r2_by_removal_point.png")

    early = ppt[ppt["point_index"] < EARLY_K]
    late = ppt[(ppt["point_index"] >= EARLY_K) & ppt["model_predicts"]]
    summary = {
        "script": SCRIPT_VERSION,
        "mean_r2_first_4": float(early["r2"].mean()),
        "mean_r2_points_5_10": float(late["r2"].mean()) if len(late) else None,
    }
    with open(out_dir / "per_point_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Per-point R² (mean over 3 attacks) ===")
    for pct, r2 in ppt.groupby("removal_pct")["r2"].mean().items():
        print(f"  {int(pct):3d}% removal: R²={r2:.3f}")
    print(f"\nMean R² points 1–{EARLY_K}: {summary['mean_r2_first_4']:.3f}")
    if summary["mean_r2_points_5_10"] is not None:
        print(f"Mean R² points 5–10: {summary['mean_r2_points_5_10']:.3f}")
    print(f"\nSaved {out_dir / 'r2_by_removal_point.png'}")


if __name__ == "__main__":
    main()
