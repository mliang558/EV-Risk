#!/usr/bin/env python3
"""
West-test breakdown tables for papers:
  - Per removal % (5, 10, 20, 30, …) × attack (B / R)
  - Per window radius (10 / 30 / 80 km) × attack
  - Per attack (same as enhancement table heads)

Usage:
  PYTHONPATH=. python -u step7_west_breakdown.py \\
    --checkpoint pyg/train_gat_global_L5_ext_br/best_model.pt \\
    --global-features-csv results_gnn/global_extended/window_global_features.csv \\
    --windows-meta windows_meta.csv \\
    --out results_gnn/west_breakdown_GAT_L5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import PCT_REMOVAL, parse_attacks
from gnn.edge_utils import sanitize_graph_list
from gnn.global_graph_features import ensure_global_features_for_eval
from gnn.metrics import curve_metrics, metrics_by_attack, split_attack_heads
from gnn.models import ResilienceGNN, resilience_kwargs_from_ckpt
from gnn.train_utils import match_target_to_pred
from step7_evaluate_gnn import load_model, predict_all

RADIUS_KM_ORDER = (10.0, 30.0, 80.0)
STRATUM_BY_RADIUS = {10.0: "urban", 30.0: "suburban", 80.0: "rural"}
RADIUS_BY_STRATUM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}
PCT_FOCUS_DEFAULT = (5, 10, 20, 30)


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def per_removal_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attacks: tuple[str, ...],
    k: int,
    pct_focus: tuple[int, ...],
) -> pd.DataFrame:
    t_parts = split_attack_heads(y_true, attacks, k)
    p_parts = split_attack_heads(y_pred, attacks, k)
    rows = []
    for attack in attacks:
        yt, yp = t_parts[attack], p_parts[attack]
        for i in range(k):
            pct = int(PCT_REMOVAL[i])
            rows.append(
                {
                    "attack": attack,
                    "removal_pct": pct,
                    "point_index": i,
                    "r2": float(r2_score(yt[:, i], yp[:, i])),
                    "in_focus": pct in pct_focus,
                }
            )
    return pd.DataFrame(rows)


def radius_label(r: float, stratum: str) -> float:
    if not np.isnan(r) and r > 0:
        return float(r)
    return float(RADIUS_BY_STRATUM.get(str(stratum).lower(), np.nan))


def per_radius_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    radii: np.ndarray,
    attacks: tuple[str, ...],
    k: int,
) -> pd.DataFrame:
    rows = []
    for r_km in RADIUS_KM_ORDER:
        mask = np.isclose(radii, r_km, rtol=0, atol=0.5)
        n = int(mask.sum())
        if n < 5:
            continue
        overall = curve_metrics(y_true[mask], y_pred[mask], n_points=k)
        by_head = metrics_by_attack(y_true[mask], y_pred[mask], attacks=attacks, n_points=k)
        row = {
            "radius_km": r_km,
            "window_stratum": STRATUM_BY_RADIUS.get(r_km, ""),
            "n_windows": n,
            "overall_r2": overall["r2"],
        }
        for a in attacks:
            row[f"{a}_r2"] = by_head[a]["r2"]
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_removal_md(df: pd.DataFrame, pct_focus: tuple[int, ...], attacks: tuple[str, ...]) -> str:
    sub = df[df["removal_pct"].isin(pct_focus)]
    lines = [
        "| Removal % | " + " | ".join(f"{a.capitalize()} R²" for a in attacks) + " |",
        "|-----------|" + "|".join(["---"] * len(attacks)) + "|",
    ]
    for pct in pct_focus:
        row = sub[sub["removal_pct"] == pct]
        cells = []
        for a in attacks:
            v = row.loc[row["attack"] == a, "r2"]
            cells.append(f"{float(v.iloc[0]):.3f}" if len(v) else "—")
        lines.append(f"| {pct}% | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pivot_radius_md(df: pd.DataFrame, attacks: tuple[str, ...]) -> str:
    if df.empty:
        return "_No windows with enough samples per radius._"
    lines = [
        "| Radius (km) | Stratum | n | "
        + " | ".join(f"{a.capitalize()} R²" for a in attacks)
        + " | Overall R² |",
        "|-------------|---------|---|"
        + "|".join(["---"] * (len(attacks) + 1))
        + "|",
    ]
    for _, r in df.iterrows():
        head_cols = " | ".join(f"{r.get(f'{a}_r2', np.nan):.3f}" for a in attacks)
        lines.append(
            f"| {int(r['radius_km'])} | {r['window_stratum']} | {int(r['n_windows'])} | "
            f"{head_cols} | {r['overall_r2']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="West breakdown: removal %, radius, attack")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--global-features-csv", default=None)
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--out", default="results_gnn/west_breakdown")
    parser.add_argument("--pct-focus", default="5,10,20,30", help="Comma-separated removal %%")
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pct_focus = tuple(int(x.strip()) for x in args.pct_focus.split(",") if x.strip())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = _torch_load(ckpt_path, "cpu")
    attacks = tuple(ckpt.get("attacks", parse_attacks("no-capacity")))

    bundle = _torch_load(Path(args.dataset).resolve(), "cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs, show_progress=False)
    split_masks = bundle.get("split_masks", {})
    train_mask = split_masks.get("east_west_train", np.ones(len(graphs), bool))
    test_mask = split_masks.get("east_west_test", np.zeros(len(graphs), bool))

    model = load_model(ckpt_path, device)
    gdim = int(getattr(model, "global_dim", 0) or ckpt.get("global_dim", 0))
    ensure_global_features_for_eval(
        graphs,
        gdim,
        train_mask,
        data_dir,
        ckpt=ckpt,
        csv_path=args.global_features_csv,
        verbose=True,
    )

    wmeta = pd.read_csv(Path(args.windows_meta).resolve())
    ok_ids = set(wmeta.loc[wmeta["n_nodes"] >= args.n_min, "window_id"].astype(str))
    meta_idx = wmeta.set_index("window_id")

    test_graphs = []
    radii = []
    for i, g in enumerate(graphs):
        if not test_mask[i]:
            continue
        wid = str(getattr(g, "window_id", ""))
        if wid not in ok_ids:
            continue
        if wid in meta_idx.index:
            row = meta_idx.loc[wid]
            r = float(row.get("radius_km", np.nan))
            s = str(row.get("window_stratum", ""))
        else:
            r, s = np.nan, ""
        radii.append(radius_label(r, s))
        g.radius_km = radii[-1]
        test_graphs.append(g)

    k = int(getattr(model, "n_out_points", ckpt.get("curve_points", 10)))
    y_true, y_pred, _ = predict_all(model, test_graphs, device)
    radii = np.array(radii, dtype=float)

    rem_df = per_removal_table(y_true, y_pred, attacks, k, pct_focus)
    rem_df.to_csv(out_dir / "per_removal_pct.csv", index=False)

    rad_df = per_radius_table(y_true, y_pred, radii, attacks, k)
    rad_df.to_csv(out_dir / "per_radius.csv", index=False)

    by_attack = metrics_by_attack(y_true, y_pred, attacks=attacks, n_points=k)
    overall = curve_metrics(y_true, y_pred, n_points=k)

    md_parts = [
        f"# West breakdown — `{ckpt_path.name}`",
        "",
        "## Per attack (full curve, all removal points pooled per head)",
        "",
        "| Attack | R² |",
        "|--------|-----|",
    ]
    for a in attacks:
        md_parts.append(f"| {a.capitalize()} | {by_attack[a]['r2']:.3f} |")
    md_parts.append(f"| **Overall** | **{overall['r2']:.3f}** |")
    md_parts.append("")
    md_parts.append(f"## Per removal % (one R² per point, West test, n={len(test_graphs)})")
    md_parts.append("")
    md_parts.append(pivot_removal_md(rem_df, pct_focus, attacks))
    md_parts.append("")
    md_parts.append("## Per window size (10 / 30 / 80 km)")
    md_parts.append("")
    md_parts.append(pivot_radius_md(rad_df, attacks))

    md_path = out_dir / "west_breakdown_tables.md"
    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")

    report = {
        "checkpoint": str(ckpt_path),
        "n_test": len(test_graphs),
        "attacks": list(attacks),
        "overall_r2": overall["r2"],
        "by_attack": {a: by_attack[a]["r2"] for a in attacks},
        "per_removal_focus": rem_df[rem_df["in_focus"]].to_dict(orient="records"),
        "per_radius": rad_df.to_dict(orient="records"),
    }
    with open(out_dir / "west_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Per removal % ===")
    for pct in pct_focus:
        sub = rem_df[rem_df["removal_pct"] == pct]
        parts = "  ".join(f"{r['attack']}={r['r2']:.3f}" for _, r in sub.iterrows())
        print(f"  {pct:3d}%  {parts}")
    print("\n=== Per radius (km) ===")
    if len(rad_df):
        print(rad_df.to_string(index=False))
    else:
        print("  (no radius groups with n>=5)")
    print(f"\nSaved -> {md_path}")


if __name__ == "__main__":
    main()
