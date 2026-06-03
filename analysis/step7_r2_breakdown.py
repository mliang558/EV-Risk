#!/usr/bin/env python3
"""
West-test R² breakdown (per removal %, per radius, GAT vs MLP).

Run from step4_gpu root:
  PYTHONPATH=. python -u step7_r2_breakdown.py \\
    --dataset pyg/pyg_dataset.pt \\
    --gat-checkpoint pyg/train_gat/best_model.pt \\
    --mlp-checkpoint pyg/train_mlp/best_model.pt \\
    --windows-meta windows_meta.csv \\
    --out results_gnn/r2_breakdown
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

N_CURVE = 10
PCT_REMOVAL = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
ATTACKS = ("betweenness", "capacity", "random")
RADIUS_BY_STRATUM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}
STRATA_ORDER = ["urban", "suburban", "rural"]

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
    gdim = int(ckpt.get("global_dim", 0))
    try:
        model = ResilienceGNN(**kw, n_out_points=k, global_dim=gdim)
    except TypeError:
        model = ResilienceGNN(**kw)
        k = 10
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if not hasattr(model, "n_out_points"):
        model.n_out_points = k
    return model, k, ckpt.get("model_type", "gat")


def prepare_graphs(graphs, model_type: str) -> None:
    try:
        from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges, sanitize_graph_list

        sanitize_graph_list(graphs, show_progress=False)
        for g in graphs:
            if model_type == "gat":
                sanitize_graph_edges(g)
            ensure_graph_level_y(g)
    except Exception:
        pass


@torch.no_grad()
def predict(model, graphs, device) -> np.ndarray:
    prepare_graphs(graphs, getattr(model, "model_type", "gat"))
    bs = 1 if getattr(model, "model_type", "") == "gat" else 64
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ps = []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        ps.append(pred.cpu().numpy())
    return np.vstack(ps)


def per_point_table(y_true_30: np.ndarray, y_pred: np.ndarray, k_model: int) -> pd.DataFrame:
    rows = []
    for a_idx, attack in enumerate(ATTACKS):
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


def per_point_by_stratum(
    y_true_30: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k_model: int
) -> pd.DataFrame:
    rows = []
    for g in STRATA_ORDER:
        mask = groups == g
        if mask.sum() < 5:
            continue
        yt_w = y_true_30[mask]
        yp_w = y_pred[mask]
        for i, pct in enumerate(PCT_REMOVAL):
            if i >= k_model:
                break
            cols_t = [yt_w[:, a * N_CURVE + i] for a in range(3)]
            cols_p = [yp_w[:, a * k_model + i] for a in range(3)]
            yt = np.concatenate(cols_t)
            yp = np.concatenate(cols_p)
            rows.append(
                {
                    "window_stratum": g,
                    "radius_km": RADIUS_BY_STRATUM.get(g, np.nan),
                    "removal_pct": pct,
                    "r2": float(r2_score(yt, yp)),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "n_windows": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_per_point(df: pd.DataFrame, out_path: Path, title: str) -> None:
    colors = {"betweenness": "#e74c3c", "capacity": "#3498db", "random": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for attack, sub in df.groupby("attack"):
        sub = sub.sort_values("removal_pct")
        m = sub["model_predicts"] if "model_predicts" in sub.columns else sub["r2"].notna()
        ax.plot(
            sub.loc[m, "removal_pct"],
            sub.loc[m, "r2"],
            "o-",
            label=attack,
            color=colors.get(attack, "gray"),
            lw=2,
        )
    valid = df[df["model_predicts"]] if "model_predicts" in df.columns else df[df["r2"].notna()]
    if len(valid):
        mean_df = valid.groupby("removal_pct", as_index=False)["r2"].mean()
        ax.plot(mean_df["removal_pct"], mean_df["r2"], "k--", marker="s", label="mean (3 attacks)", lw=1.5)
    ax.set_xlabel("Node removal (%)")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_radius_per_point(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"urban": "#2ecc71", "suburban": "#3498db", "rural": "#e67e22"}
    for g, sub in df.groupby("window_stratum"):
        sub = sub.sort_values("removal_pct")
        ax.plot(
            sub["removal_pct"],
            sub["r2"],
            "o-",
            label=f"{g} ({RADIUS_BY_STRATUM.get(g, 0):.0f} km)",
            color=colors.get(g, "gray"),
            lw=2,
        )
    ax.set_xlabel("Removal %")
    ax.set_ylabel("R² (3 attacks pooled)")
    ax.set_title("R² by window radius × removal %")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gat_vs_mlp_per_point(compare: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = compare["removal_pct"].values
    ax.plot(x, compare["gat_r2"], "o-", color="#2980b9", lw=2, label="GAT")
    ax.plot(x, compare["mlp_r2"], "s-", color="#95a5a6", lw=2, label="MLP")
    ax.fill_between(x, compare["gat_r2"], compare["mlp_r2"], alpha=0.15, color="#7f8c8d")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Node removal (%)")
    ax.set_ylabel("R² (mean over 3 attacks)")
    ax.set_title("GAT vs MLP — R² at each removal % (West)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    for _, row in compare.iterrows():
        d = row["gat_r2"] - row["mlp_r2"]
        ax.annotate(f"{d:+.2f}", (row["removal_pct"], max(row["gat_r2"], row["mlp_r2"]) + 0.02), fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="R² breakdown: per %, per radius, GAT vs MLP")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--gat-checkpoint", required=True)
    parser.add_argument("--mlp-checkpoint", default=None)
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--out", default="results_gnn/r2_breakdown")
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wmeta = pd.read_csv(Path(args.windows_meta).resolve())
    ok_ids = set(wmeta.loc[wmeta["n_nodes"] >= args.n_min, "window_id"].astype(str))
    meta_stratum = wmeta.set_index("window_id")["window_stratum"]

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    test_mask = bundle.get("split_masks", {}).get("east_west_test", np.zeros(len(graphs), bool))
    test_graphs = []
    for i, g in enumerate(graphs):
        if not test_mask[i]:
            continue
        wid = str(getattr(g, "window_id", ""))
        if wid not in ok_ids:
            continue
        test_graphs.append(g)

    print(f"West test | n>={args.n_min} | n={len(test_graphs)}", flush=True)
    y_true_30 = np.vstack([g.y.view(-1).numpy()[:30] for g in test_graphs])
    groups = np.array(
        [str(meta_stratum.get(str(getattr(g, "window_id", "")), getattr(g, "window_stratum", ""))) for g in test_graphs],
        dtype=object,
    )

    # --- GAT ---
    gat_model, k_gat, _ = load_model(Path(args.gat_checkpoint).resolve(), device)
    y_pred_gat = predict(gat_model, test_graphs, device)
    gat_ppt = per_point_table(y_true_30, y_pred_gat, k_gat)
    gat_ppt.to_csv(out_dir / "1_per_removal_pct_gat.csv", index=False)
    plot_per_point(gat_ppt, out_dir / "1_r2_by_removal_pct_gat.png", "GAT — R² by removal % (West)")

    gat_mean = gat_ppt[gat_ppt["model_predicts"]].groupby("removal_pct", as_index=False).agg(
        r2=("r2", "mean"), mae=("mae", "mean")
    )
    gat_mean.to_csv(out_dir / "1_per_removal_pct_mean.csv", index=False)

    # --- Per radius ---
    gat_radius = per_point_by_stratum(y_true_30, y_pred_gat, groups, k_gat)
    gat_radius.to_csv(out_dir / "2_per_radius_per_point.csv", index=False)
    plot_radius_per_point(gat_radius, out_dir / "2_r2_by_radius_per_point.png")

    radius_summary = []
    for g in STRATA_ORDER:
        m = groups == g
        if m.sum() < 5:
            continue
        radius_summary.append(
            {
                "window_stratum": g,
                "radius_km": RADIUS_BY_STRATUM[g],
                "n_windows": int(m.sum()),
                "r2_overall_30d": float(r2_score(y_true_30[m].ravel(), y_pred_gat[m].ravel())),
            }
        )
    pd.DataFrame(radius_summary).to_csv(out_dir / "2_per_radius_overall.csv", index=False)

    # --- GAT vs MLP per point ---
    compare_rows = []
    if args.mlp_checkpoint:
        mlp_model, k_mlp, _ = load_model(Path(args.mlp_checkpoint).resolve(), device)
        y_pred_mlp = predict(mlp_model, test_graphs, device)
        mlp_ppt = per_point_table(y_true_30, y_pred_mlp, k_mlp)
        mlp_ppt.to_csv(out_dir / "3_per_removal_pct_mlp.csv", index=False)
        plot_per_point(mlp_ppt, out_dir / "3_r2_by_removal_pct_mlp.png", "MLP — R² by removal % (West)")

        gat_m = gat_ppt[gat_ppt["model_predicts"]].groupby("removal_pct")["r2"].mean()
        mlp_m = mlp_ppt[mlp_ppt["model_predicts"]].groupby("removal_pct")["r2"].mean()
        for pct in PCT_REMOVAL[: min(k_gat, k_mlp)]:
            compare_rows.append(
                {
                    "removal_pct": pct,
                    "gat_r2": float(gat_m.get(pct, np.nan)),
                    "mlp_r2": float(mlp_m.get(pct, np.nan)),
                    "delta_gat_minus_mlp": float(gat_m.get(pct, np.nan) - mlp_m.get(pct, np.nan)),
                }
            )
        compare_df = pd.DataFrame(compare_rows)
        compare_df.to_csv(out_dir / "3_gat_vs_mlp_per_point.csv", index=False)
        plot_gat_vs_mlp_per_point(compare_df, out_dir / "3_gat_vs_mlp_per_point.png")

    report = {
        "n_test": len(test_graphs),
        "n_min": args.n_min,
        "best_radius_overall": max(radius_summary, key=lambda r: r["r2_overall_30d"]) if radius_summary else None,
        "best_removal_pct_gat": gat_mean.loc[gat_mean["r2"].idxmax()].to_dict() if len(gat_mean) else None,
        "worst_removal_pct_gat": gat_mean.loc[gat_mean["r2"].idxmin()].to_dict() if len(gat_mean) else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== 1) GAT R² by removal % (mean over 3 attacks) ===")
    for _, r in gat_mean.iterrows():
        print(f"  {int(r['removal_pct']):3d}%  R²={r['r2']:.3f}  MAE={r['mae']:.3f}")

    print("\n=== 2) GAT overall R² by radius ===")
    for r in radius_summary:
        print(f"  {r['window_stratum']:9s} {r['radius_km']:.0f} km  R²={r['r2_overall_30d']:.3f}  n={r['n_windows']}")

    if compare_rows:
        print("\n=== 3) GAT vs MLP (mean R² per removal %) ===")
        for r in compare_rows:
            print(f"  {int(r['removal_pct']):3d}%  GAT={r['gat_r2']:.3f}  MLP={r['mlp_r2']:.3f}  Δ={r['delta_gat_minus_mlp']:+.3f}")

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
