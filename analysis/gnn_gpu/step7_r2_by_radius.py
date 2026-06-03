#!/usr/bin/env python3
"""
Sensitivity: West-test R² by window radius (10 / 30 / 80 km).

Uses windows_meta window_stratum or radius_km. Reports B / R / Overall per radius.

Usage:
  PYTHONPATH=. python -u step7_r2_by_radius.py \\
    --dataset pyg/pyg_dataset.pt \\
    --checkpoint pyg/train_gat_global_L5_ext_br/best_model.pt \\
    --global-features-csv results_gnn/global_extended/window_global_features.csv \\
    --windows-meta windows_meta.csv --out results_gnn/r2_by_radius_L5_ext
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

from gnn.constants import ATTACK_ORDER
from gnn.edge_utils import sanitize_graph_list
from gnn.metrics import curve_metrics, metrics_by_attack
from gnn.models import ResilienceGNN, model_uses_edge_attr, resilience_kwargs_from_ckpt
from gnn.train_utils import match_target_to_pred

RADIUS_KM_ORDER = (10.0, 30.0, 80.0)
STRATUM_BY_RADIUS = {10.0: "urban", 30.0: "suburban", 80.0: "rural"}
RADIUS_BY_STRATUM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(ckpt_path: Path, device: torch.device) -> ResilienceGNN:
    ckpt = _torch_load(ckpt_path, "cpu")
    model = ResilienceGNN(**resilience_kwargs_from_ckpt(ckpt))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_graphs(model, graphs, device, attacks):
    bs = 1 if model_uses_edge_attr(getattr(model, "model_type", "")) else 64
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    ys, ps, ids, radii, strata = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _, _ = model(batch)
        tgt = batch.y
        if tgt.dim() == 1:
            tgt = tgt.view(pred.size(0), -1)
        tgt = match_target_to_pred(tgt, pred, attacks)
        ys.append(tgt.cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(list(batch.window_id))
        if hasattr(batch, "radius_km"):
            radii.extend([float(x) for x in batch.radius_km.view(-1).tolist()])
        else:
            radii.extend([np.nan] * batch.num_graphs)
        if hasattr(batch, "window_stratum"):
            strata.extend(list(batch.window_stratum))
        else:
            strata.extend([""] * batch.num_graphs)
    return np.vstack(ys), np.vstack(ps), ids, np.array(radii), np.array(strata, dtype=object)


def attach_global(
    graphs,
    data_dir: Path,
    train_mask: np.ndarray,
    gcsv: Path | None,
    *,
    global_dim: int,
    ckpt: dict,
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
        csv_path=gcsv,
        verbose=True,
    )


def radius_label(r: float, stratum: str) -> float:
    if not np.isnan(r) and r > 0:
        return float(r)
    return float(RADIUS_BY_STRATUM.get(str(stratum).lower(), np.nan))


def main() -> None:
    parser = argparse.ArgumentParser(description="R² by window radius (West test)")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--global-features-csv", default=None)
    parser.add_argument("--out", default="results_gnn/r2_by_radius")
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    wmeta = pd.read_csv(Path(args.windows_meta).resolve())
    ok_ids = set(wmeta.loc[wmeta["n_nodes"] >= args.n_min, "window_id"].astype(str))
    meta_idx = wmeta.set_index("window_id")

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs, show_progress=False)
    split_masks = bundle.get("split_masks", {})
    train_mask = split_masks.get("east_west_train", np.ones(len(graphs), bool))
    test_mask = split_masks.get("east_west_test", np.zeros(len(graphs), bool))

    gcsv = Path(args.global_features_csv) if args.global_features_csv else None
    if gcsv is None:
        for cand in (
            data_dir / "results_gnn/global_extended/window_global_features.csv",
            data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
        ):
            if cand.is_file():
                gcsv = cand
                break
    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = _torch_load(ckpt_path, "cpu")
    gdim = int(ckpt.get("global_dim", 0))
    attach_global(graphs, data_dir, train_mask, gcsv, global_dim=gdim, ckpt=ckpt)

    test_graphs = []
    for i, g in enumerate(graphs):
        if not test_mask[i]:
            continue
        wid = str(getattr(g, "window_id", ""))
        if wid not in ok_ids:
            continue
        if wid in meta_idx.index:
            row = meta_idx.loc[wid]
            g.radius_km = float(row.get("radius_km", np.nan))
            g.window_stratum = str(row.get("window_stratum", ""))
        test_graphs.append(g)

    print(f"West test | n_nodes>={args.n_min} | n={len(test_graphs)}", flush=True)

    model = load_model(Path(args.checkpoint).resolve(), device)
    attacks = getattr(model, "attacks", ("betweenness", "random"))
    k = getattr(model, "n_out_points", 10)

    y_true, y_pred, ids, radii_g, strata_g = predict_graphs(model, test_graphs, device, attacks)

    radii = []
    for wid, rg, sg in zip(ids, radii_g, strata_g):
        if wid in meta_idx.index:
            r = float(meta_idx.loc[wid].get("radius_km", np.nan))
            s = str(meta_idx.loc[wid].get("window_stratum", sg))
        else:
            r, s = float(rg), str(sg)
        radii.append(radius_label(r, s))
    radii = np.array(radii, dtype=float)

    rows = []
    for r_km in RADIUS_KM_ORDER:
        mask = np.isclose(radii, r_km, rtol=0, atol=0.5)
        n = int(mask.sum())
        if n < 5:
            continue
        overall = curve_metrics(y_true[mask], y_pred[mask], n_points=k)
        by_head = metrics_by_attack(y_true[mask], y_pred[mask], attacks=attacks, n_points=k)
        rows.append(
            {
                "radius_km": r_km,
                "window_stratum": STRATUM_BY_RADIUS.get(r_km, ""),
                "n_windows": n,
                "overall_r2": overall["r2"],
                "overall_mae": overall["mae"],
                **{f"{a}_r2": by_head[a]["r2"] for a in attacks},
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "r2_by_radius_summary.csv", index=False)

    md = [
        "| Radius (km) | Stratum | n | Betweenness R² | Random R² | Overall R² |",
        "|-------------|---------|---|----------------|-----------|------------|",
    ]
    for _, r in summary_df.iterrows():
        md.append(
            f"| {int(r['radius_km'])} | {r['window_stratum']} | {int(r['n_windows'])} | "
            f"{r.get('betweenness_r2', np.nan):.3f} | {r.get('random_r2', np.nan):.3f} | "
            f"{r['overall_r2']:.3f} |"
        )
    (out_dir / "r2_by_radius_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if not args.no_plots and len(summary_df) > 0:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(summary_df))
            ax.bar(x, summary_df["overall_r2"], color=["#2ecc71", "#3498db", "#e67e22"][: len(summary_df)])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(r)} km" for r in summary_df["radius_km"]])
            ax.set_ylabel("Overall R² (West)")
            ax.set_ylim(0, 1)
            ax.set_title("Sensitivity by window radius")
            fig.tight_layout()
            fig.savefig(out_dir / "r2_by_radius_bar.png", dpi=150)
            plt.close(fig)
        except ImportError:
            pass

    report = {"n_min": args.n_min, "n_test": len(test_graphs), "attacks": list(attacks), "by_radius": rows}
    with open(out_dir / "r2_by_radius_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== R² by radius (West) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
