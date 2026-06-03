#!/usr/bin/env python3
"""
SHAP analysis for LightGBM main model (East train, West test).

Outputs:
  - shap_global_importance.csv
  - shap_attack_importance.csv  (betweenness vs random, aggregated over 10 points)
  - shap_summary.md
  - optional PNG bar plots

Usage:
  PYTHONPATH=. python -u step7_lgb_shap_analysis.py \
    --data-dir . \
    --out results_gnn/lgb_shap_main \
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
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import N_CURVE_POINTS, parse_attacks, y_cols_for_attacks
from gnn.local_window_features import LOCAL_FEATURE_NAMES, build_local_feature_table
from step7_window_lgb_baseline import build_feature_table, make_model


def _attack_output_slices(attacks: tuple[str, ...]) -> dict[str, list[int]]:
    out = {}
    start = 0
    for a in attacks:
        out[a] = list(range(start, start + N_CURVE_POINTS))
        start += N_CURVE_POINTS
    return out


def _compute_shap_for_multioutput(model, x_df: pd.DataFrame) -> np.ndarray:
    """
    Return abs SHAP values: (n_outputs, n_samples, n_features).
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("Install shap first: pip install shap") from e

    vals = []
    for est in model.estimators_:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(x_df)
        arr = np.asarray(sv)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        vals.append(np.abs(arr))
    return np.stack(vals, axis=0)


def _save_bar_png(df: pd.DataFrame, col: str, out_png: Path, title: str, top_k: int = 18) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    d = df.sort_values(col, ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["feature"][::-1], d[col][::-1], color="#4e79a7")
    ax.set_title(title)
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM SHAP analysis for paper")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="results_gnn/lgb_shap")
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
    parser.add_argument("--max-shap-samples", type=int, default=2000)
    args = parser.parse_args()

    attacks = parse_attacks(args.attacks)
    if set(attacks) != {"betweenness", "random"}:
        print(f"[SHAP] WARN: expected no-capacity (B+R), got {attacks}", flush=True)

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
    Y_train = Y[train_mask]
    X_train_df = pd.DataFrame(X_train, columns=x_cols)
    X_test_df = pd.DataFrame(X_test, columns=x_cols)

    model = make_model(args.model, args.n_estimators, args.max_depth, args.num_leaves, args.n_jobs)
    print(
        f"[SHAP] Fit LightGBM: train={len(X_train_df)} test={len(X_test_df)} F={len(x_cols)} "
        f"targets={len(y_cols)}",
        flush=True,
    )
    model.fit(X_train_df, Y_train)

    n_sample = min(args.max_shap_samples, len(X_test_df))
    shap_input = X_test_df.sample(n=n_sample, random_state=42) if n_sample < len(X_test_df) else X_test_df
    print(f"[SHAP] Computing SHAP on n={len(shap_input)} West samples...", flush=True)
    shap_abs = _compute_shap_for_multioutput(model, shap_input)
    # shap_abs shape: (n_outputs, n_samples, n_features)

    global_imp = shap_abs.mean(axis=(0, 1))
    global_df = pd.DataFrame({"feature": x_cols, "mean_abs_shap": global_imp})
    global_df = global_df.sort_values("mean_abs_shap", ascending=False)
    global_df.to_csv(out_dir / "shap_global_importance.csv", index=False)

    slices = _attack_output_slices(attacks)
    rows = []
    for a, idxs in slices.items():
        imp = shap_abs[idxs].mean(axis=(0, 1))
        for f, v in zip(x_cols, imp):
            rows.append({"attack": a, "feature": f, "mean_abs_shap": float(v)})
    attack_df = pd.DataFrame(rows).sort_values(["attack", "mean_abs_shap"], ascending=[True, False])
    attack_df.to_csv(out_dir / "shap_attack_importance.csv", index=False)

    compare_df = (
        attack_df.pivot_table(index="feature", columns="attack", values="mean_abs_shap", aggfunc="mean")
        .fillna(0.0)
        .reset_index()
    )
    if "betweenness" in compare_df.columns and "random" in compare_df.columns:
        compare_df["delta_b_minus_r"] = compare_df["betweenness"] - compare_df["random"]
    compare_df.to_csv(out_dir / "shap_attack_importance_compare.csv", index=False)

    _save_bar_png(global_df, "mean_abs_shap", out_dir / "shap_global_top18.png", "LightGBM SHAP global importance")
    if "betweenness" in compare_df.columns:
        _save_bar_png(compare_df, "betweenness", out_dir / "shap_betweenness_top18.png", "SHAP importance for betweenness")
    if "random" in compare_df.columns:
        _save_bar_png(compare_df, "random", out_dir / "shap_random_top18.png", "SHAP importance for random")
    if "delta_b_minus_r" in compare_df.columns:
        _save_bar_png(
            compare_df.assign(abs_delta=np.abs(compare_df["delta_b_minus_r"])),
            "abs_delta",
            out_dir / "shap_b_vs_r_delta_top18.png",
            "Feature difference |SHAP_B - SHAP_R|",
        )

    md = [
        "# LightGBM SHAP Summary",
        "",
        f"- Feature set: `{args.feature_set}` (`{args.global_feature_set}` global set)",
        f"- Attacks: `{','.join(attacks)}`",
        f"- SHAP sample size (West): `{len(shap_input)}`",
        "",
        "## Top 18 global features",
        "",
        "| Rank | Feature | mean(|SHAP|) |",
        "|------|---------|--------------|",
    ]
    for i, (_, r) in enumerate(global_df.head(18).iterrows(), start=1):
        md.append(f"| {i} | {r['feature']} | {r['mean_abs_shap']:.6f} |")
    if "delta_b_minus_r" in compare_df.columns:
        top_delta = compare_df.reindex(compare_df["delta_b_minus_r"].abs().sort_values(ascending=False).index).head(10)
        md += [
            "",
            "## Features with largest B vs R importance difference",
            "",
            "| Rank | Feature | SHAP(B) | SHAP(R) | Delta (B-R) |",
            "|------|---------|---------|---------|-------------|",
        ]
        for i, (_, r) in enumerate(top_delta.iterrows(), start=1):
            md.append(
                f"| {i} | {r['feature']} | {r.get('betweenness', 0.0):.6f} | "
                f"{r.get('random', 0.0):.6f} | {r['delta_b_minus_r']:.6f} |"
            )
    (out_dir / "shap_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = {
        "feature_set": args.feature_set,
        "global_feature_set": args.global_feature_set,
        "attacks": list(attacks),
        "n_train": int(len(X_train_df)),
        "n_test": int(len(X_test_df)),
        "n_shap_samples": int(len(shap_input)),
        "top10_global_features": global_df.head(10).to_dict(orient="records"),
    }
    with open(out_dir / "shap_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SHAP] Saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
