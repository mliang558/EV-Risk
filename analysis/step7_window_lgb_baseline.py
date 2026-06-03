#!/usr/bin/env python3
"""
Quick check: do global graph features predict attack curves? (LightGBM default)

East train / West test (center_lon > -100), same as GAT.

Usage:
  cd step4_gpu
  PYTHONPATH=. python -u step7_window_lgb_baseline.py \\
    --data-dir . --windows-meta windows_meta.csv

  # fastest sanity check (3 AUC targets):
  PYTHONPATH=. python -u step7_window_lgb_baseline.py --target auc3 --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import Y_ALL_COLS, parse_attacks, y_cols_for_attacks
from gnn.local_window_features import LOCAL_FEATURE_NAMES, build_local_feature_table
from gnn.metrics import metrics_by_attack
from gnn.global_graph_features import (
    GLOBAL_FEATURE_NAMES,
    global_features_from_npz,
    subgraph_path_for_window,
)

PCT_REMOVAL = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
AUC_COLS = ["y_betweenness_auc", "y_capacity_auc", "y_random_auc"]


def build_feature_table(
    df: pd.DataFrame,
    data_dir: Path,
    cache_csv: Path | None,
    *,
    reuse_only: bool = False,
) -> pd.DataFrame:
    if cache_csv and cache_csv.is_file() and reuse_only:
        print(f"[LGB] Load cached features: {cache_csv}", flush=True)
        return pd.read_csv(cache_csv)
    print(f"[LGB] Building global features for {len(df)} windows...", flush=True)
    rows = []
    it = df.itertuples(index=False)
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="Global feats", unit="win", file=sys.stderr)
    for row in it:
        wid = str(row.window_id)
        sp = subgraph_path_for_window(data_dir, wid)
        if sp is None:
            continue
        f = global_features_from_npz(sp)
        f["window_id"] = wid
        rows.append(f)
    feat_df = pd.DataFrame(rows)
    if cache_csv is not None:
        feat_df.to_csv(cache_csv, index=False)
        print(f"[LGB] Cached -> {cache_csv}", flush=True)
    return feat_df


def make_model(
    name: str,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
    n_jobs: int,
):
    if name in ("lightgbm", "lgb"):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("Install: pip install lightgbm") from e
        base = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
            verbose=-1,
        )
    elif name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor

        base = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return MultiOutputRegressor(base, n_jobs=n_jobs)


def per_point_r2_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attacks: tuple[str, ...] = ("betweenness", "capacity", "random"),
) -> pd.DataFrame:
    from gnn.attack_targets import select_attacks_numpy
    from gnn.constants import ATTACK_ORDER, N_CURVE_POINTS

    yt = select_attacks_numpy(y_true, attacks, N_CURVE_POINTS)
    yp = select_attacks_numpy(y_pred, attacks, N_CURVE_POINTS)
    k = yt.shape[1] // len(attacks)
    rows = []
    for i, pct in enumerate(PCT_REMOVAL[:k]):
        cols_t = [yt[:, a * k + i] for a in range(len(attacks))]
        cols_p = [yp[:, a * k + i] for a in range(len(attacks))]
        rows.append(
            {
                "removal_pct": pct,
                "r2": float(r2_score(np.concatenate(cols_t), np.concatenate(cols_p))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM + global graph features (quick validation)")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="results_gnn/lgb_baseline")
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument("--target", choices=["curve30", "auc3"], default="curve30")
    parser.add_argument("--model", choices=["lightgbm", "lgb", "gbm"], default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel fits across 30 outputs")
    parser.add_argument("--fast", action="store_true", help="n_estimators=80, for quick run")
    parser.add_argument("--reuse-features", action="store_true", help="Load window_global_features.csv if present")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--attacks",
        default="all",
        help="Label columns: all | no-capacity | betweenness,random",
    )
    parser.add_argument(
        "--feature-set",
        choices=["global", "local", "both"],
        default="global",
        help="global=15 graph stats | local=node aggregates | both=concat",
    )
    args = parser.parse_args()
    attack_tuple = parse_attacks(args.attacks)

    n_est = args.n_estimators or (80 if args.fast else 200)

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / args.windows_meta)
    manifest = pd.read_csv(data_dir / "manifest.csv")

    labels_path = Path(args.labels) if args.labels else data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)

    df = labels.copy()
    if "subgraph_path" not in df.columns and "subgraph_path" in manifest.columns:
        df = df.merge(manifest[["window_id", "subgraph_path"]], on="window_id", how="left")
    if "subgraph_path" not in df.columns:
        df["subgraph_path"] = df["window_id"].astype(str).map(lambda w: f"subgraphs/{w}.npz")

    meta_cols = [c for c in ("window_id", "center_lon", "center_lat", "window_stratum", "radius_km", "n_nodes") if c in meta.columns]
    if len(meta_cols) > 1:
        df = df.merge(meta[meta_cols], on="window_id", how="left", suffixes=("", "_meta"))
    if args.limit:
        df = df.head(args.limit)

    merged = df
    x_cols: list[str] = []

    if args.feature_set in ("global", "both"):
        cache_g = out_dir / "window_global_features.csv"
        if args.reuse_features and not cache_g.is_file():
            for cand in (
                data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
                data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
            ):
                if cand.is_file():
                    import shutil

                    shutil.copy(cand, cache_g)
                    break
        global_df = build_feature_table(df, data_dir, cache_g, reuse_only=args.reuse_features)
        overlap = [c for c in global_df.columns if c in merged.columns and c != "window_id"]
        merged = merged.merge(global_df.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")
        x_cols.extend([c for c in GLOBAL_FEATURE_NAMES if c in merged.columns])

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

    if args.target == "auc3":
        y_cols = [c for c in AUC_COLS if c in merged.columns]
        if len(y_cols) < 3:
            raise SystemExit("Need y_*_auc in labels (step4 --also-auc) or use --target curve30")
    else:
        y_cols = [c for c in y_cols_for_attacks(attack_tuple) if c in merged.columns]
        if len(y_cols) != len(attack_tuple) * 10:
            y_cols = [c for c in Y_ALL_COLS if c in merged.columns]

    if not x_cols:
        raise SystemExit(f"No features for --feature-set {args.feature_set}")

    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    train_mask = merged["center_lon"].astype(float).values > args.lon_threshold

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[~train_mask])
    Y_train, Y_test = Y[train_mask], Y[~train_mask]
    X_train_df = pd.DataFrame(X_train, columns=x_cols)
    X_test_df = pd.DataFrame(X_test, columns=x_cols)

    print(
        f"[LGB] feature_set={args.feature_set} | Train {train_mask.sum()} | Test {(~train_mask).sum()} | "
        f"F={len(x_cols)} | targets={len(y_cols)} | trees={n_est}",
        flush=True,
    )
    model = make_model(args.model, n_est, args.max_depth, args.num_leaves, args.n_jobs)
    print("[LGB] Fitting...", flush=True)
    model.fit(X_train_df, Y_train)
    pred_test = model.predict(X_test_df)

    by_head = metrics_by_attack(Y_test, pred_test, attacks=attack_tuple)
    report = {
        "model": args.model,
        "feature_set": args.feature_set,
        "attacks": list(attack_tuple),
        "target": args.target,
        "n_features": len(x_cols),
        "feature_names": x_cols,
        "n_estimators": n_est,
        "test_r2": float(r2_score(Y_test.ravel(), pred_test.ravel())),
        "test_mae": float(mean_absolute_error(Y_test.ravel(), pred_test.ravel())),
        "test_betweenness_r2": float(by_head["betweenness"]["r2"]),
        "test_random_r2": float(by_head["random"]["r2"]),
        "betweenness": by_head["betweenness"],
        "random": by_head["random"],
    }
    if args.target == "curve30" and Y_test.shape[1] == len(attack_tuple) * 10:
        ppt = per_point_r2_table(Y_test, pred_test, attacks=attack_tuple)
        ppt.to_csv(out_dir / "per_point_r2.csv", index=False)
        report["per_point_mean_r2"] = float(ppt["r2"].mean())

    with open(out_dir / "lgb_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== LightGBM ({args.feature_set}) West ===")
    print(f"  test_r2  = {report['test_r2']:.4f}")
    print(f"  test_mae = {report['test_mae']:.4f}")
    print(f"  B_r2={report['test_betweenness_r2']:.4f}  R_r2={report['test_random_r2']:.4f}")
    if "per_point_mean_r2" in report:
        print(f"  per-point mean R² = {report['per_point_mean_r2']:.4f}")
    print(f"\nSaved -> {out_dir / 'lgb_report.json'}")
    print("If test_r2 >> 0.5: global features have signal (compare to GAT).")


if __name__ == "__main__":
    main()
