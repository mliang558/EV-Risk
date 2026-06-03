#!/usr/bin/env python3
"""
Optuna hyperparameter search for LightGBM + window global features.

Same setup as step7_window_lgb_baseline.py:
  East train (center_lon > -100) / West test.

Usage (step4_gpu root):
  pip install lightgbm optuna

  PYTHONPATH=. python -u step7_window_lgb_optuna.py \\
    --data-dir . \\
    --reuse-features \\
    --out results_gnn/lgb_optuna \\
    --n-trials 50

  # Tune on 15% holdout from East (less leaky); final West R² in report:
  PYTHONPATH=. python -u step7_window_lgb_optuna.py --objective val --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step7_window_lgb_baseline import (  # noqa: E402
    build_feature_table,
    per_point_r2_table,
)
from gnn.constants import parse_attacks, y_cols_for_attacks  # noqa: E402
from gnn.global_graph_features import GLOBAL_FEATURE_NAMES  # noqa: E402


def load_merged_table(data_dir: Path, windows_meta: str, labels: Path | None, limit: int | None):
    meta = pd.read_csv(data_dir / windows_meta)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    labels_path = labels or data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    lab = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)

    df = lab.copy()
    if "subgraph_path" not in df.columns and "subgraph_path" in manifest.columns:
        df = df.merge(manifest[["window_id", "subgraph_path"]], on="window_id", how="left")
    if "subgraph_path" not in df.columns:
        df["subgraph_path"] = df["window_id"].astype(str).map(lambda w: f"subgraphs/{w}.npz")

    meta_cols = [
        c
        for c in ("window_id", "center_lon", "center_lat", "window_stratum", "radius_km", "n_nodes")
        if c in meta.columns
    ]
    if len(meta_cols) > 1:
        df = df.merge(meta[meta_cols], on="window_id", how="left", suffixes=("", "_meta"))
    if limit:
        df = df.head(limit)
    return df, meta


def prepare_xy(
    merged: pd.DataFrame,
    lon_threshold: float,
    y_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    x_cols = [c for c in GLOBAL_FEATURE_NAMES if c in merged.columns]
    if not x_cols:
        exclude = {"window_id", "subgraph_path", "center_lon", "center_lat", "window_stratum"}
        exclude |= set(y_cols) | {c for c in merged.columns if c.startswith("y_")}
        x_cols = [
            c
            for c in merged.columns
            if c not in exclude and merged[c].dtype in ("float64", "float32", "int64", "int32")
        ]

    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    train_mask = merged["center_lon"].astype(float).values > lon_threshold

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[~train_mask])
    Y_train, Y_test = Y[train_mask], Y[~train_mask]
    return X_train, X_test, Y_train, Y_test, x_cols, scaler


def _as_feature_frame(X: np.ndarray, feature_names: list[str] | None) -> pd.DataFrame | np.ndarray:
    """Keep train/predict consistent for LGBM sklearn API (avoids feature-name warnings)."""
    if feature_names is None:
        return X
    return pd.DataFrame(X, columns=list(feature_names))


def fit_lgbm_predict(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_ev: np.ndarray,
    params: dict,
    n_jobs: int,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    from lightgbm import LGBMRegressor

    X_tr_in = _as_feature_frame(X_tr, feature_names)
    X_ev_in = _as_feature_frame(X_ev, feature_names)
    base = LGBMRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )
    model = MultiOutputRegressor(base, n_jobs=n_jobs)
    model.fit(X_tr_in, Y_tr)
    return model.predict(X_ev_in)


def split_train_val(X_train: np.ndarray, Y_train: np.ndarray, val_fraction: float, seed: int):
    n = X_train.shape[0]
    idx = np.arange(n)
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed)
    return X_train[tr], Y_train[tr], X_train[va], Y_train[va]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tune LightGBM (global window features)")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="results_gnn/lgb_optuna")
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel targets in MultiOutputRegressor")
    parser.add_argument(
        "--objective",
        choices=["west", "val"],
        default="west",
        help="west=maximize West test R² (same as baseline); val=tune on East holdout",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--study-name", default="lgb_global_curve30")
    parser.add_argument("--attacks", default="all", help="all | no-capacity")
    args = parser.parse_args()
    attack_tuple = parse_attacks(args.attacks)

    try:
        import optuna
    except ImportError as e:
        raise SystemExit("Install: pip install optuna lightgbm") from e

    try:
        from lightgbm import LGBMRegressor  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install: pip install lightgbm") from e

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Path(args.labels) if args.labels else None
    df, _meta = load_merged_table(data_dir, args.windows_meta, labels_path, args.limit)

    cache_csv = out_dir / "window_global_features.csv"
    if args.reuse_features and not cache_csv.is_file():
        for cand in (
            data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
            data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
        ):
            if cand.is_file():
                import shutil

                shutil.copy(cand, cache_csv)
                print(f"[Optuna] Reuse features from {cand}", flush=True)
                break

    feat_df = build_feature_table(df, data_dir, cache_csv, reuse_only=args.reuse_features)
    overlap = [c for c in feat_df.columns if c in df.columns and c != "window_id"]
    merged = df.merge(feat_df.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")

    y_cols = [c for c in y_cols_for_attacks(attack_tuple) if c in merged.columns]
    if len(y_cols) != len(attack_tuple) * 10:
        raise SystemExit(f"Expected {len(attack_tuple)*10} curve columns, got {len(y_cols)}")

    X_train, X_test, Y_train, Y_test, x_cols, _scaler = prepare_xy(
        merged, args.lon_threshold, y_cols
    )
    print(
        f"[Optuna] Train {X_train.shape[0]} | West test {X_test.shape[0]} | F={len(x_cols)} | "
        f"objective={args.objective} | trials={args.n_trials}",
        flush=True,
    )

    if args.objective == "val":
        X_tr, Y_tr, X_va, Y_va = split_train_val(X_train, Y_train, args.val_fraction, args.seed)
    else:
        X_tr, Y_tr, X_va, Y_va = X_train, Y_train, X_test, Y_test

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]),
        }
        pred = fit_lgbm_predict(X_tr, Y_tr, X_va, params, args.n_jobs, x_cols)
        return float(r2_score(Y_va.ravel(), pred.ravel()))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        storage=f"sqlite:///{out_dir / 'optuna_study.db'}",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)
    best_params["colsample_bytree"] = 0.8

    print("\n=== Optuna best trial ===", flush=True)
    print(f"  best_objective R² = {study.best_value:.4f}", flush=True)
    print(f"  params = {best_params}", flush=True)

    pred_west = fit_lgbm_predict(X_train, Y_train, X_test, best_params, args.n_jobs, x_cols)
    west_r2 = float(r2_score(Y_test.ravel(), pred_west.ravel()))
    west_mae = float(mean_absolute_error(Y_test.ravel(), pred_west.ravel()))
    ppt = per_point_r2_table(Y_test, pred_west, attacks=attack_tuple)
    ppt.to_csv(out_dir / "per_point_r2.csv", index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "trials.csv", index=False)

    report = {
        "model": "lightgbm",
        "attacks": list(attack_tuple),
        "optuna_objective": args.objective,
        "best_trial_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_features": len(x_cols),
        "feature_names": x_cols,
        "west_test_r2": west_r2,
        "west_test_mae": west_mae,
        "per_point_mean_r2": float(ppt["r2"].mean()),
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    with open(out_dir / "lgb_optuna_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== West test (retrain on full East with best params) ===")
    print(f"  test_r2  = {west_r2:.4f}")
    print(f"  test_mae = {west_mae:.4f}")
    print(f"  per-point mean R² = {report['per_point_mean_r2']:.4f}")
    print(f"\nSaved -> {out_dir}/lgb_optuna_report.json")
    print(f"       -> {out_dir}/best_params.json")
    print(f"       -> {out_dir}/trials.csv")


if __name__ == "__main__":
    main()
