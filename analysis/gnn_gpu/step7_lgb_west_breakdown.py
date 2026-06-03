#!/usr/bin/env python3
"""
Train LightGBM (East) and report West breakdown tables for paper:
  - Per removal % (focus: 5/10/20/30)
  - Per window size (10 / 30 / 80 km)
  - Per attack (betweenness / random)

Usage:
  PYTHONPATH=. python -u step7_lgb_west_breakdown.py \
    --data-dir . \
    --out results_gnn/lgb_breakdown_main \
    --global-feature-set extended \
    --attacks no-capacity \
    --reuse-features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.attack_targets import select_attacks_numpy
from gnn.constants import ATTACK_ORDER, N_CURVE_POINTS, PCT_REMOVAL, parse_attacks, y_cols_for_attacks
from gnn.metrics import curve_metrics, metrics_by_attack
from gnn.local_window_features import LOCAL_FEATURE_NAMES, build_local_feature_table
from step7_window_lgb_baseline import build_feature_table, make_model

RADIUS_KM_ORDER = (10.0, 30.0, 80.0)
STRATUM_BY_RADIUS = {10.0: "urban", 30.0: "suburban", 80.0: "rural"}
RADIUS_BY_STRATUM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}


def radius_label(r: float, stratum: str) -> float:
    if not np.isnan(r) and r > 0:
        return float(r)
    return float(RADIUS_BY_STRATUM.get(str(stratum).lower(), np.nan))


def per_removal_attack_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attacks: tuple[str, ...],
) -> pd.DataFrame:
    yt = select_attacks_numpy(y_true, attacks, N_CURVE_POINTS)
    yp = select_attacks_numpy(y_pred, attacks, N_CURVE_POINTS)
    rows = []
    for a_idx, attack in enumerate(attacks):
        for i, pct in enumerate(PCT_REMOVAL):
            rows.append(
                {
                    "attack": attack,
                    "point_index": i,
                    "removal_pct": int(pct),
                    "r2": float(r2_score(yt[:, a_idx * N_CURVE_POINTS + i], yp[:, a_idx * N_CURVE_POINTS + i])),
                }
            )
    return pd.DataFrame(rows)


def per_radius_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    radii: np.ndarray,
    attacks: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for r_km in RADIUS_KM_ORDER:
        mask = np.isclose(radii, r_km, rtol=0, atol=0.5)
        n = int(mask.sum())
        if n < 5:
            continue
        overall = curve_metrics(y_true[mask], y_pred[mask], n_points=N_CURVE_POINTS)
        by_head = metrics_by_attack(y_true[mask], y_pred[mask], attacks=attacks, n_points=N_CURVE_POINTS)
        row = {
            "radius_km": r_km,
            "window_stratum": STRATUM_BY_RADIUS.get(r_km, ""),
            "n_windows": n,
            "overall_r2": overall["r2"],
            "overall_mae": overall["mae"],
        }
        for a in attacks:
            row[f"{a}_r2"] = by_head[a]["r2"]
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_removal_md(df: pd.DataFrame, attacks: tuple[str, ...], pct_focus: tuple[int, ...]) -> str:
    lines = [
        "| Removal % | " + " | ".join(f"{a.capitalize()} R²" for a in attacks) + " |",
        "|-----------|" + "|".join(["---"] * len(attacks)) + "|",
    ]
    sub = df[df["removal_pct"].isin(pct_focus)]
    for pct in pct_focus:
        row = sub[sub["removal_pct"] == pct]
        vals = []
        for a in attacks:
            v = row.loc[row["attack"] == a, "r2"]
            vals.append(f"{float(v.iloc[0]):.3f}" if len(v) else "—")
        lines.append(f"| {pct}% | " + " | ".join(vals) + " |")
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
    parser = argparse.ArgumentParser(description="LightGBM West breakdown for paper tables")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="results_gnn/lgb_breakdown")
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument("--model", choices=["lightgbm", "lgb", "gbm"], default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--feature-set", choices=["global", "local", "both"], default="global")
    parser.add_argument("--global-feature-set", choices=["base", "extended"], default="extended")
    parser.add_argument("--global-fast", action="store_true")
    parser.add_argument("--attacks", default="no-capacity")
    parser.add_argument("--pct-focus", default="5,10,20,30")
    parser.add_argument("--n-min", type=int, default=15)
    args = parser.parse_args()

    attacks = parse_attacks(args.attacks)
    if any(a not in ATTACK_ORDER for a in attacks):
        raise SystemExit(f"Unsupported attacks {attacks}; expected subset of {ATTACK_ORDER}")
    pct_focus = tuple(int(x.strip()) for x in args.pct_focus.split(",") if x.strip())

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / args.windows_meta)
    labels_path = Path(args.labels) if args.labels else data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)

    df = labels.copy()
    meta_cols = [c for c in ("window_id", "center_lon", "center_lat", "window_stratum", "radius_km", "n_nodes") if c in meta.columns]
    if len(meta_cols) > 1:
        df = df.merge(meta[meta_cols], on="window_id", how="left", suffixes=("", "_meta"))

    merged = df
    x_cols: list[str] = []
    if args.feature_set in ("global", "both"):
        cache_g = out_dir / "window_global_features.csv"
        global_df = build_feature_table(
            df,
            data_dir,
            cache_g,
            reuse_only=args.reuse_features,
            feature_set=args.global_feature_set,
            fast=args.global_fast,
        )
        overlap = [c for c in global_df.columns if c in merged.columns and c != "window_id"]
        merged = merged.merge(global_df.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")
        x_cols.extend([c for c in global_df.columns if c != "window_id"])
    if args.feature_set in ("local", "both"):
        local_df = build_local_feature_table(
            merged["window_id"].astype(str).tolist(),
            data_dir,
            cache_csv=out_dir / "window_local_features.csv",
            windows_meta=meta,
        )
        overlap = [c for c in local_df.columns if c in merged.columns and c != "window_id"]
        merged = merged.merge(local_df.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")
        x_cols.extend([c for c in LOCAL_FEATURE_NAMES if c in merged.columns])
    x_cols = list(dict.fromkeys(x_cols))

    y_cols = [c for c in y_cols_for_attacks(attacks) if c in merged.columns]
    if len(y_cols) != len(attacks) * N_CURVE_POINTS:
        raise SystemExit(f"Need {len(attacks)*N_CURVE_POINTS} labels, got {len(y_cols)}")
    if not x_cols:
        raise SystemExit("No feature columns found.")

    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    train_mask = merged["center_lon"].astype(float).values > args.lon_threshold
    test_mask = ~train_mask

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    Y_train, Y_test = Y[train_mask], Y[test_mask]

    model = make_model(args.model, args.n_estimators, args.max_depth, args.num_leaves, args.n_jobs)
    print(
        f"[LGB breakdown] Train {train_mask.sum()} | Test {test_mask.sum()} | F={len(x_cols)} | "
        f"targets={len(y_cols)}",
        flush=True,
    )
    X_train_df = pd.DataFrame(X_train, columns=x_cols)
    X_test_df = pd.DataFrame(X_test, columns=x_cols)
    model.fit(X_train_df, Y_train)
    pred_train = model.predict(X_train_df)
    pred_test = model.predict(X_test_df)

    test_df = merged.loc[test_mask].copy()
    if "n_nodes" in test_df.columns:
        test_keep = test_df["n_nodes"].fillna(0).astype(float).values >= args.n_min
        Y_test = Y_test[test_keep]
        pred_test = pred_test[test_keep]
        test_df = test_df.loc[test_keep].reset_index(drop=True)

    if "radius_km" in test_df.columns:
        rs = test_df["radius_km"].values
    else:
        rs = np.full(len(test_df), np.nan)
    if "window_stratum" in test_df.columns:
        ss = test_df["window_stratum"].values
    else:
        ss = np.array([""] * len(test_df), dtype=object)
    radii = np.array(
        [
            radius_label(
                float(r) if pd.notna(r) else np.nan,
                str(s) if pd.notna(s) else "",
            )
            for r, s in zip(rs, ss)
        ],
        dtype=float,
    )

    rem_df = per_removal_attack_table(Y_test, pred_test, attacks)
    rem_df.to_csv(out_dir / "per_removal_pct.csv", index=False)

    rad_df = per_radius_table(Y_test, pred_test, radii, attacks)
    rad_df.to_csv(out_dir / "per_radius.csv", index=False)

    by_attack = metrics_by_attack(Y_test, pred_test, attacks=attacks, n_points=N_CURVE_POINTS)
    overall = curve_metrics(Y_test, pred_test, n_points=N_CURVE_POINTS)
    train_by_attack = metrics_by_attack(Y_train, pred_train, attacks=attacks, n_points=N_CURVE_POINTS)
    train_overall = curve_metrics(Y_train, pred_train, n_points=N_CURVE_POINTS)

    md_parts = [
        "# West breakdown — LightGBM",
        "",
        f"Feature set: `{args.feature_set}` | global: `{args.global_feature_set}` | attacks: `{','.join(attacks)}`",
        "",
        "## Per attack (full curve pooled per head)",
        "",
        "| Attack | R² |",
        "|--------|-----|",
    ]
    for a in attacks:
        md_parts.append(f"| {a.capitalize()} | {by_attack[a]['r2']:.3f} |")
    md_parts.append(f"| **Overall** | **{overall['r2']:.3f}** |")
    md_parts.append("")
    md_parts.append("## Per removal % (focus)")
    md_parts.append("")
    md_parts.append(pivot_removal_md(rem_df, attacks, pct_focus))
    md_parts.append("")
    md_parts.append("## Per window size (10 / 30 / 80 km)")
    md_parts.append("")
    md_parts.append(pivot_radius_md(rad_df, attacks))
    md_parts.append("")
    md_parts.append("## Geographic Split Validation (East train -> West test)")
    md_parts.append("")
    md_parts.append("| Split | " + " | ".join(f"{a.capitalize()} R²" for a in attacks) + " | Overall R² |")
    md_parts.append("|-------|" + "|".join(["---"] * (len(attacks) + 1)) + "|")
    tr_cells = " | ".join(f"{train_by_attack[a]['r2']:.3f}" for a in attacks)
    te_cells = " | ".join(f"{by_attack[a]['r2']:.3f}" for a in attacks)
    md_parts.append(f"| East (train) | {tr_cells} | {train_overall['r2']:.3f} |")
    md_parts.append(f"| West (test) | {te_cells} | {overall['r2']:.3f} |")
    (out_dir / "west_breakdown_tables.md").write_text("\n".join(md_parts) + "\n", encoding="utf-8")

    report = {
        "model": "lightgbm",
        "feature_set": args.feature_set,
        "global_feature_set": args.global_feature_set,
        "attacks": list(attacks),
        "n_test": int(len(Y_test)),
        "n_train": int(len(Y_train)),
        "overall_r2": float(overall["r2"]),
        "overall_mae": float(overall["mae"]),
        "train_overall_r2": float(train_overall["r2"]),
        "train_overall_mae": float(train_overall["mae"]),
        "by_attack_r2": {a: float(by_attack[a]["r2"]) for a in attacks},
        "train_by_attack_r2": {a: float(train_by_attack[a]["r2"]) for a in attacks},
    }
    with open(out_dir / "lgb_breakdown_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Per removal focus ===")
    for pct in pct_focus:
        sub = rem_df[rem_df["removal_pct"] == pct]
        parts = "  ".join(f"{r['attack']}={r['r2']:.3f}" for _, r in sub.iterrows())
        print(f"  {pct:3d}%  {parts}")
    print("\n=== Per radius ===")
    if len(rad_df):
        print(rad_df.to_string(index=False))
    else:
        print("  (no radius groups with n>=5)")
    print("\n=== Geographic split validation ===")
    print(f"  East(train): overall R²={train_overall['r2']:.3f}")
    print(f"  West(test):  overall R²={overall['r2']:.3f}")
    print(f"\nSaved -> {out_dir / 'west_breakdown_tables.md'}")


if __name__ == "__main__":
    main()
