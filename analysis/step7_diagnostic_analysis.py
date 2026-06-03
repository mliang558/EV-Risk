#!/usr/bin/env python3
"""
Post-hoc diagnostics on West test (Step 7 extension).

1. Pred vs true scatter + error-by-magnitude bins
2. R² / MAE by window_stratum (urban / suburban / rural)
3. GAT vs MLP R² comparison (requires MLP checkpoint from step6 --model mlp)

Usage (GPU server, after Step 6/7):
  cd /opt/data_repo/mliang_work/step4_gpu
  python step7_diagnostic_analysis.py \\
    --dataset pyg/pyg_dataset.pt \\
    --gat-checkpoint pyg/train_gat/best_model.pt \\
    --mlp-checkpoint pyg/train_mlp/best_model.pt \\
    --out results_gnn/diagnostics
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
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges, sanitize_graph_list
from gnn import constants as _gnn_const

N_CURVE_POINTS = int(getattr(_gnn_const, "N_CURVE_POINTS", 10))
PCT_REMOVAL = list(
    getattr(_gnn_const, "PCT_REMOVAL", [5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
)
EARLY_CURVE_POINTS = int(getattr(_gnn_const, "EARLY_CURVE_POINTS", 4))
from gnn.metrics import (
    curve_metrics,
    metrics_by_group,
    per_point_metrics_table,
    per_point_table_vs_full_labels,
    split_three_heads,
)
from gnn.train_utils import match_target_to_pred
from gnn.models import ResilienceGNN

STRATA_ORDER = ["urban", "suburban", "rural"]
HEADS = ["betweenness", "capacity", "random"]


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(ckpt_path: Path, device: torch.device) -> ResilienceGNN:
    ckpt = _torch_load(ckpt_path, device)
    model_type = ckpt.get("model_type", "gat")
    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=int(ckpt.get("hidden", 32)),
        gat_heads=int(ckpt.get("gat_heads", 4)),
        edge_dim=1 if model_type == "gat" else 0,
        model_type=model_type,
        n_out_points=int(ckpt.get("curve_points", 10)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_all(model, graphs, device, batch_size: int = 64):
    for g in graphs:
        if getattr(model, "model_type", "") == "gat":
            sanitize_graph_edges(g)
        ensure_graph_level_y(g)

    bs = 1 if getattr(model, "model_type", "") == "gat" else batch_size
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ys, ps, ids, strata = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        elif tgt.dim() == 3 and tgt.size(1) == 1:
            tgt = tgt.squeeze(1)
        tgt = match_target_to_pred(tgt, pred)
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(list(batch.window_id))
        if hasattr(batch, "window_stratum"):
            strata.extend(list(batch.window_stratum))
        else:
            strata.extend([""] * int(batch.num_graphs))
    return np.vstack(ys), np.vstack(ps), ids, np.array(strata, dtype=object)


def plot_scatter_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Scatter colored by |error|; highlights where predictions are worst."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    err = np.abs(yp - yt)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    sc = ax.scatter(yt, yp, c=err, cmap="hot_r", alpha=0.35, s=10, vmin=0, vmax=np.quantile(err, 0.95))
    lim = max(float(yt.max()), float(yp.max()), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    plt.colorbar(sc, ax=ax, label="|error|")
    m = curve_metrics(y_true, y_pred)
    ax.set_xlabel("True relative loss")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nR²={m['r2']:.3f} MAE={m['mae']:.3f}")

    ax = axes[1]
    ax.scatter(yt, err, alpha=0.25, s=8, c="steelblue")
    ax.set_xlabel("True relative loss")
    ax.set_ylabel("|Predicted − True|")
    ax.set_title("Absolute error vs true magnitude")
    ax.axhline(m["mae"], color="red", ls="--", lw=1, label=f"MAE={m['mae']:.3f}")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_by_true_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    """MAE and count per decile of true value (where model struggles)."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    err = np.abs(yp - yt)

    edges = np.quantile(yt, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(yt.min(), yt.max() + 1e-6, n_bins + 1)

    bin_idx = np.digitize(yt, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        lo, hi = edges[b], edges[b + 1]
        rows.append(
            {
                "bin": b,
                "true_lo": lo,
                "true_hi": hi,
                "n": int(mask.sum()),
                "mae": float(err[mask].mean()),
                "mean_true": float(yt[mask].mean()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax1.bar(x - 0.2, df["mae"], width=0.4, label="MAE", color="coral")
    ax1.set_ylabel("MAE")
    ax1.set_xlabel("True value bin (relative loss)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r['true_lo']:.2f}–{r['true_hi']:.2f}" for _, r in df.iterrows()], rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, df["n"], width=0.4, alpha=0.4, color="gray", label="count")
    ax2.set_ylabel("# points")
    ax1.set_title("Error vs true-value range (all 30-dim points)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return df


def plot_stratum_r2_bar(stratum_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if stratum_df.empty:
        return
    df = stratum_df.copy()
    df["group"] = pd.Categorical(df["group"], categories=STRATA_ORDER, ordered=True)
    df = df.sort_values("group")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    ax.bar(x, df["r2"], color=["#2ecc71", "#3498db", "#e67e22"][: len(df)])
    ax.set_xticks(x)
    ax.set_xticklabels(df["group"].astype(str))
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for xi, (_, r) in enumerate(df.iterrows()):
        ax.text(xi, r["r2"] + 0.02, f"n={int(r['n'])}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_point_r2(df: pd.DataFrame, out_path: Path, k_focus: int = EARLY_CURVE_POINTS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"betweenness": "#e74c3c", "capacity": "#3498db", "random": "#2ecc71"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("r2", "mae")):
        for attack, sub in df.groupby("attack"):
            sub = sub.sort_values("point_index")
            mask = sub["r2"].notna() if metric == "r2" else sub["mae"].notna()
            ax.plot(
                sub.loc[mask, "removal_pct"],
                sub.loc[mask, metric],
                "o-",
                label=attack,
                color=colors.get(attack, "gray"),
                lw=2,
            )
        valid = df[df["r2"].notna()] if metric == "r2" else df
        mean_df = valid.groupby("removal_pct", as_index=False)[metric].mean()
        if len(mean_df):
            ax.plot(mean_df["removal_pct"], mean_df[metric], "k--", marker="s", label="mean", lw=1.5)
        if k_focus <= N_CURVE_POINTS:
            ax.axvline(PCT_REMOVAL[k_focus - 1], color="gray", ls=":", lw=1.5)
        ax.set_xlabel("Node removal (%)")
        ax.set_ylabel("R²" if metric == "r2" else "MAE")
        ax.set_title(f"{metric.upper()} by removal point")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric == "r2":
            ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gat_vs_mlp(compare_df: pd.DataFrame, out_path: Path) -> None:
    if compare_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(compare_df))
    w = 0.35
    ax.bar(x - w / 2, compare_df["gat_r2"], width=w, label="GAT", color="#2980b9")
    ax.bar(x + w / 2, compare_df["mlp_r2"], width=w, label="MLP (no graph)", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["metric"])
    ax.set_ylabel("R² (West test)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("GAT vs MLP — similar R² ⇒ limited graph signal")
    for i, row in compare_df.iterrows():
        delta = row["gat_r2"] - row["mlp_r2"]
        ax.text(i, max(row["gat_r2"], row["mlp_r2"]) + 0.02, f"Δ={delta:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_model_diag(
    name: str,
    ckpt_path: Path,
    test_graphs: list,
    meta_df: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
) -> dict:
    model = load_model(ckpt_path, device)
    y_true, y_pred, ids, strata = predict_all(model, test_graphs, device, batch_size)

    sub = meta_df[meta_df["window_id"].isin(ids)].copy()
    if "window_stratum" in sub.columns:
        strata_meta = sub.set_index("window_id")["window_stratum"].reindex(ids).fillna("").values
    else:
        strata_meta = strata
    # prefer meta table stratum
    groups = np.where(
        pd.Series(strata_meta).astype(str).str.len() > 0,
        strata_meta,
        strata,
    )

    prefix = out_dir / name
    prefix.mkdir(parents=True, exist_ok=True)

    k_model = int(getattr(model, "n_out_points", 10))
    id_set = set(ids)
    y_true_30 = np.vstack(
        [g.y.view(-1).numpy()[:30] for g in test_graphs if str(getattr(g, "window_id", "")) in id_set]
    )
    if len(y_true_30) == len(y_pred):
        if k_model == N_CURVE_POINTS and y_pred.shape[1] == 30:
            ppt = per_point_metrics_table(y_true_30, y_pred, n_points=10)
        else:
            ppt = per_point_table_vs_full_labels(y_true_30, y_pred, k_model)
        ppt.to_csv(prefix / "r2_mae_per_point.csv", index=False)
        plot_per_point_r2(ppt, prefix / "r2_by_removal_point.png")

    plot_scatter_pred_vs_true(y_true, y_pred, prefix / "scatter_pred_vs_true.png", f"{name.upper()} West test")
    bin_df = plot_error_by_true_bins(y_true, y_pred, prefix / "error_by_true_bin.png")
    bin_df.to_csv(prefix / "error_by_true_bin.csv", index=False)

    overall = curve_metrics(y_true, y_pred)
    stratum_df = metrics_by_group(y_true, y_pred, groups)
    stratum_df.to_csv(prefix / "metrics_by_stratum.csv", index=False)
    plot_stratum_r2_bar(stratum_df, prefix / "r2_by_stratum.png", f"{name.upper()} R² by stratum")

    head_rows = []
    yb_t, yc_t, yr_t = split_three_heads(y_true)
    yb_p, yc_p, yr_p = split_three_heads(y_pred)
    for head, yt, yp in zip(HEADS, [yb_t, yc_t, yr_t], [yb_p, yc_p, yr_p]):
        head_rows.append({"head": head, **curve_metrics(yt, yp)})
        sd = metrics_by_group(yt, yp, groups)
        sd["head"] = head
        sd.to_csv(prefix / f"metrics_by_stratum_{head}.csv", index=False)

    pd.DataFrame(
        {
            "window_id": ids,
            "window_stratum": groups,
            **{f"y_true_{i}": y_true[:, i] for i in range(y_true.shape[1])},
            **{f"y_pred_{i}": y_pred[:, i] for i in range(y_pred.shape[1])},
        }
    ).to_csv(prefix / "west_predictions_full.csv", index=False)

    report = {
        "model": name,
        "checkpoint": str(ckpt_path),
        "n_test": len(ids),
        "overall": overall,
        "by_stratum": stratum_df.to_dict(orient="records"),
        "by_head": head_rows,
    }
    with open(prefix / "diagnostic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"=== {name.upper()} West test ===")
    print(f"  Overall R²={overall['r2']:.4f} MAE={overall['mae']:.4f}")
    for _, r in stratum_df.iterrows():
        print(f"  {r['group']}: R²={r['r2']:.4f} MAE={r['mae']:.4f} (n={int(r['n'])})")

    return {"y_true": y_true, "y_pred": y_pred, "overall": overall, "stratum_df": stratum_df, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7 diagnostics: scatter, stratum, GAT vs MLP")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--gat-checkpoint", default="pyg/train_gat/best_model.pt")
    parser.add_argument("--mlp-checkpoint", default=None, help="From step6 --model mlp")
    parser.add_argument("--out", default="results_gnn/diagnostics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs)
    meta_df = bundle["meta"]
    split_masks = bundle.get("split_masks", {})
    test_mask = split_masks.get("east_west_test", np.zeros(len(graphs), bool))
    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]

    gat_path = Path(args.gat_checkpoint).resolve()
    gat_out = run_model_diag("gat", gat_path, test_graphs, meta_df, device, out_dir, args.batch_size)

    compare_rows = [
        {
            "metric": "overall",
            "gat_r2": gat_out["overall"]["r2"],
            "mlp_r2": np.nan,
            "gat_mae": gat_out["overall"]["mae"],
            "mlp_mae": np.nan,
        }
    ]
    for head in HEADS:
        h = next(x for x in gat_out["report"]["by_head"] if x["head"] == head)
        compare_rows.append(
            {"metric": head, "gat_r2": h["r2"], "mlp_r2": np.nan, "gat_mae": h["mae"], "mlp_mae": np.nan}
        )

    if args.mlp_checkpoint:
        mlp_path = Path(args.mlp_checkpoint).resolve()
        if mlp_path.exists():
            mlp_out = run_model_diag("mlp", mlp_path, test_graphs, meta_df, device, out_dir, args.batch_size)
            compare_rows[0]["mlp_r2"] = mlp_out["overall"]["r2"]
            compare_rows[0]["mlp_mae"] = mlp_out["overall"]["mae"]
            for i, head in enumerate(HEADS, start=1):
                h = next(x for x in mlp_out["report"]["by_head"] if x["head"] == head)
                compare_rows[i]["mlp_r2"] = h["r2"]
                compare_rows[i]["mlp_mae"] = h["mae"]

            merged_stratum = gat_out["stratum_df"].merge(
                mlp_out["stratum_df"],
                on="group",
                suffixes=("_gat", "_mlp"),
                how="outer",
            )
            merged_stratum.to_csv(out_dir / "stratum_gat_vs_mlp.csv", index=False)
        else:
            print(f"[warn] MLP checkpoint not found: {mlp_path}")
    else:
        print("[info] Pass --mlp-checkpoint after: python step6_train_gnn.py --model mlp --out pyg/train_mlp")

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_dir / "gat_vs_mlp_comparison.csv", index=False)
    if compare_df["mlp_r2"].notna().any():
        plot_gat_vs_mlp(compare_df, out_dir / "gat_vs_mlp_r2.png")

    summary = {
        "gat": gat_out["report"],
        "mlp_note": "Train MLP with step6 --model mlp, then re-run with --mlp-checkpoint",
        "compare": compare_rows,
    }
    with open(out_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDiagnostics -> {out_dir}")
    print("  scatter_pred_vs_true.png, error_by_true_bin.png, r2_by_stratum.png")
    if args.mlp_checkpoint:
        print("  gat_vs_mlp_r2.png, gat_vs_mlp_comparison.csv")


if __name__ == "__main__":
    main()
