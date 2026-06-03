#!/usr/bin/env python3
"""
West-test R² / MAE by sampling radius (urban 10km / suburban 30km / rural 80km).
Only windows with n_nodes >= --n-min (default 15).

Usage:
  cd /opt/data_repo/mliang_work/step4_gpu
  python step7_r2_by_radius.py \\
    --dataset pyg/pyg_dataset.pt \\
    --checkpoint pyg/train_gat/best_model.pt \\
    --windows-meta windows_meta.csv \\
    --out results_gnn/r2_by_radius \\
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
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PCT_REMOVAL = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
RADIUS_BY_STRATUM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}
STRATA_ORDER = ["urban", "suburban", "rural"]


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
        return target.view(target.size(0), 3, 10)[:, :, :k].reshape(target.size(0), 3 * k)
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
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_graphs(model, graphs, device):
    try:
        from gnn.edge_utils import ensure_graph_level_y, sanitize_graph_edges, sanitize_graph_list

        sanitize_graph_list(graphs, show_progress=False)
        for g in graphs:
            if getattr(model, "model_type", "") == "gat":
                sanitize_graph_edges(g)
            ensure_graph_level_y(g)
    except Exception:
        pass

    bs = 1 if getattr(model, "model_type", "") == "gat" else 64
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ys, ps, ids, strata = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        tgt = match_target_to_pred(tgt, pred)
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(list(batch.window_id))
        strata.extend(list(getattr(batch, "window_stratum", [""] * batch.num_graphs)))
    return np.vstack(ys), np.vstack(ps), ids, np.array(strata, dtype=object)


def metrics_for_group(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return {"r2": float(r2_score(yt, yp)), "mae": float(mean_absolute_error(yt, yp)), "n": len(y_true)}


def per_point_r2_by_stratum(y_true_30, y_pred, groups, k_model: int = 10) -> pd.DataFrame:
    from sklearn.metrics import r2_score

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
            cols_t = [yt_w[:, a * 10 + i] for a in range(3)]
            cols_p = [yp_w[:, a * k_model + i] for a in range(3)]
            yt = np.concatenate(cols_t)
            yp = np.concatenate(cols_p)
            rows.append(
                {
                    "window_stratum": g,
                    "radius_km": RADIUS_BY_STRATUM.get(g, np.nan),
                    "removal_pct": pct,
                    "r2": float(r2_score(yt, yp)),
                    "n_windows": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="R² by urban/suburban/rural radius stratum")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--out", default="results_gnn/r2_by_radius")
    parser.add_argument("--n-min", type=int, default=15, help="Only evaluate windows with >= n nodes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wmeta = pd.read_csv(Path(args.windows_meta).resolve())
    ok_ids = set(wmeta.loc[wmeta["n_nodes"] >= args.n_min, "window_id"].astype(str))

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

    print(f"West test | n_nodes>={args.n_min} | graphs={len(test_graphs)} (of {int(test_mask.sum())})", flush=True)

    model = load_model(Path(args.checkpoint).resolve(), device)
    y_matched, y_pred, ids, strata = predict_graphs(model, test_graphs, device)
    # full 30-dim labels for per-point breakdown
    y_true_30 = np.vstack([g.y.view(-1).numpy()[:30] for g in test_graphs])

    # stratum from meta if missing on graph
    meta_stratum = wmeta.set_index("window_id")["window_stratum"]
    groups = np.array(
        [str(meta_stratum.get(wid, s)) for wid, s in zip(ids, strata)],
        dtype=object,
    )

    rows = []
    for g in STRATA_ORDER:
        mask = groups == g
        n = int(mask.sum())
        if n < 5:
            continue
        m = metrics_for_group(y_true_30[mask], y_pred[mask])
        rows.append(
            {
                "window_stratum": g,
                "radius_km": RADIUS_BY_STRATUM[g],
                "n_windows": n,
                "n_nodes_mean": float(wmeta[wmeta["window_stratum"] == g]["n_nodes"].mean()),
                **m,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "r2_by_radius_summary.csv", index=False)

    ppt = per_point_r2_by_stratum(y_true_30, y_pred, groups)
    ppt.to_csv(out_dir / "r2_by_radius_per_point.csv", index=False)

    # bar chart overall R² by stratum
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["r2"], color=["#2ecc71", "#3498db", "#e67e22"][: len(summary_df)])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['window_stratum']}\n({int(r['radius_km'])} km)" for _, r in summary_df.iterrows()]
    )
    ax.set_ylabel("R² (30-dim, West)")
    ax.set_ylim(0, 1)
    ax.set_title(f"GAT West test | windows with n>={args.n_min}")
    for i, r in summary_df.iterrows():
        ax.text(i, r["r2"] + 0.02, f"n={int(r['n_windows'])}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "r2_by_radius_bar.png", dpi=150)
    plt.close(fig)

    # lines: R² vs removal % per stratum
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"urban": "#2ecc71", "suburban": "#3498db", "rural": "#e67e22"}
    for g, sub in ppt.groupby("window_stratum"):
        sub = sub.sort_values("removal_pct")
        ax.plot(sub["removal_pct"], sub["r2"], "o-", label=f"{g} ({RADIUS_BY_STRATUM[g]:.0f} km)", color=colors.get(g))
    ax.set_xlabel("Removal %")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Per-point R² by radius stratum (n>={args.n_min})")
    fig.tight_layout()
    fig.savefig(out_dir / "r2_by_radius_per_point.png", dpi=150)
    plt.close(fig)

    report = {
        "n_min": args.n_min,
        "n_test_graphs": len(test_graphs),
        "by_stratum": rows,
    }
    with open(out_dir / "r2_by_radius_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== R² by sampling radius (West, n>=%d) ===" % args.n_min)
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
