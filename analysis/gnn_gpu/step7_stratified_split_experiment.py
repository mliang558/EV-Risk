#!/usr/bin/env python3
"""
Compare LightGBM+Global under three train/test protocols:

  1. east_west          — East train / West test (longitude)
  2. stratified_stratum — National; matched urban/suburban/rural % in train & test
  3. stratified_state   — Whole states held out; test set matches stratum mix

Usage (step4_gpu):
  PYTHONPATH=. python -u step7_stratified_split_experiment.py \\
    --data-dir . \\
    --out results_gnn/stratified_split_lgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import parse_attacks, y_cols_for_attacks
from gnn.metrics import metrics_by_attack
from gnn.splits import (
    build_split_masks,
    format_stratum_report,
    resolve_train_test_masks,
    split_key_prefix,
)
from step7_window_lgb_baseline import build_feature_table, make_model

SPLITS = ("east_west", "stratified_stratum", "stratified_state")


def run_lgb_one_split(
    merged: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    attacks: tuple[str, ...],
    model_name: str,
    n_est: int,
) -> dict:
    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_mask])
    X_te = scaler.transform(X[test_mask])
    Y_tr, Y_te = Y[train_mask], Y[test_mask]
    model = make_model(model_name, n_est, 8, 31, -1)
    model.fit(pd.DataFrame(X_tr, columns=x_cols), Y_tr)
    pred = model.predict(pd.DataFrame(X_te, columns=x_cols))
    overall = {
        "r2": float(r2_score(Y_te.ravel(), pred.ravel())),
        "mae": float(mean_absolute_error(Y_te.ravel(), pred.ravel())),
    }
    by_head = metrics_by_attack(Y_te, pred, attacks=attacks)
    return {"overall": overall, "by_attack": by_head, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description="LGB split comparison (east_west vs stratified)")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--windows-meta", default="windows_meta.csv")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt", help="Optional; reuse split_masks if present")
    parser.add_argument("--out", default="results_gnn/stratified_split_lgb")
    parser.add_argument("--attacks", default="no-capacity")
    parser.add_argument("--global-feature-set", default="extended")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--lon-threshold", type=float, default=-100.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    attacks = parse_attacks(args.attacks)
    meta = pd.read_csv(data_dir / args.windows_meta)
    manifest = pd.read_csv(data_dir / "manifest.csv")

    labels_path = data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)
    merged = labels.merge(
        meta[["window_id", "center_lon", "center_lat", "window_stratum", "radius_km", "n_nodes"]],
        on="window_id",
        how="left",
    )

    cache = out_dir / "window_global_features.csv"
    gdf = build_feature_table(
        merged,
        data_dir,
        cache,
        reuse_only=cache.is_file(),
        feature_set=args.global_feature_set,
    )
    overlap = [c for c in gdf.columns if c in merged.columns and c != "window_id"]
    merged = merged.merge(gdf.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")
    x_cols = [c for c in gdf.columns if c != "window_id"]
    y_cols = [c for c in y_cols_for_attacks(attacks) if c in merged.columns]

    split_masks = None
    ds_path = Path(args.dataset)
    if ds_path.is_file():
        import torch

        try:
            bundle = torch.load(ds_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(ds_path, map_location="cpu")
        split_masks = bundle.get("split_masks")

    if split_masks is None or "stratified_stratum_train" not in split_masks:
        split_masks = build_split_masks(
            meta,
            manifest=manifest,
            lon_threshold=args.lon_threshold,
            test_frac=args.test_frac,
            seed=args.split_seed,
            include=SPLITS,
        )

    rows = []
    reports = {}
    for split in SPLITS:
        tr_meta, te_meta = resolve_train_test_masks(meta, split_masks, split, manifest=manifest)
        print(f"\n=== {split} ===\n{format_stratum_report(meta, tr_meta, te_meta)}")
        pos = {str(w): i for i, w in enumerate(meta["window_id"].astype(str))}
        tr = np.array([tr_meta[pos[str(w)]] for w in merged["window_id"].astype(str)])
        te = np.array([te_meta[pos[str(w)]] for w in merged["window_id"].astype(str)])
        res = run_lgb_one_split(merged, x_cols, y_cols, tr, te, attacks, "lightgbm", args.n_estimators)
        reports[split] = res
        for head in attacks:
            rows.append(
                {
                    "split": split,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "attack": head,
                    "r2": res["by_attack"][head]["r2"],
                    "mae": res["by_attack"][head]["mae"],
                    "overall_r2": res["overall"]["r2"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lgb_by_split.csv", index=False)

    md = [
        "# LightGBM + Global — split comparison",
        "",
        "| Split | n_train | n_test | Betweenness R² | Random R² | Overall R² |",
        "|-------|---------|--------|----------------|-----------|------------|",
    ]
    for split in SPLITS:
        r = reports[split]
        b = r["by_attack"].get("betweenness", {})
        rnd = r["by_attack"].get("random", {})
        md.append(
            f"| {split} | {r['n_train']} | {r['n_test']} | "
            f"{b.get('r2', float('nan')):.3f} | {rnd.get('r2', float('nan')):.3f} | "
            f"{r['overall']['r2']:.3f} |"
        )
    md.append("")
    md.append("## Stratum balance (train | test)")
    md.append("")
    for split in SPLITS:
        prefix = split_key_prefix(split)
        tr, te = split_masks[f"{prefix}_train"], split_masks[f"{prefix}_test"]
        md.append(f"### {split}")
        md.append("```")
        md.append(format_stratum_report(meta, tr, te))
        md.append("```")
        md.append("")

    (out_dir / "split_comparison.md").write_text("\n".join(md), encoding="utf-8")
    with open(out_dir / "split_comparison.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=float)
    print(f"\nSaved -> {out_dir / 'split_comparison.md'}")


if __name__ == "__main__":
    main()
