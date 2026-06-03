#!/usr/bin/env python3
"""
Evaluate ALL models from enhancement_manifest under stratified splits.

Does NOT overwrite existing East/West results (results_gnn/eval_train_*).
Writes new outputs under:
  results_gnn/stratified_eval/{split}/{model_slug}/test_metrics.json
  lgbm/train_lgbm_ext_br_{split}/lgb_report.json

Then builds per-split tables + a combined markdown.

Usage:
  PYTHONPATH=. python -u step7_run_stratified_splits_all_models.py --data-dir .
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import parse_attacks, y_cols_for_attacks
from gnn.mlflow_utils import MlflowTracker

try:
    from gnn.mlflow_utils import log_stratum_metrics
except ImportError:
    def log_stratum_metrics(tracker: MlflowTracker, props: dict[str, float], prefix: str) -> None:
        if not tracker.enabled:
            return
        tracker.log_metrics({f"{prefix}_{k}_pct": float(v) for k, v in props.items()})

from gnn.splits import format_stratum_report, split_key_prefix, stratum_distribution
from step7_build_comparison_table import eval_gnn_checkpoint, metrics_from_report_json
from step7_build_enhancement_report import format_md
from step7_window_lgb_baseline import build_feature_table, make_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

NEW_SPLITS = ("stratified_stratum", "stratified_state")


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()[:80]


def slug_from_ckpt(ckpt: str) -> str:
    p = Path(ckpt)
    return slug(p.parent.name if p.parent.name else p.stem)


def run_lgb_split(
    data_dir: Path,
    split: str,
    out_dir: Path,
    global_csv: Path,
    attacks: tuple[str, ...],
    n_est: int,
    *,
    skip_existing: bool = False,
) -> Path:
    """Train LGB on train_mask, eval on test_mask; save lgb_report.json."""
    from gnn.splits import resolve_train_test_masks

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "lgb_report.json"
    if skip_existing and report_path.is_file():
        return report_path

    meta = pd.read_csv(data_dir / "windows_meta.csv")
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
    gdf = pd.read_csv(global_csv)
    overlap = [c for c in gdf.columns if c in merged.columns and c != "window_id"]
    merged = merged.merge(gdf.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")
    x_cols = [c for c in gdf.columns if c != "window_id"]
    y_cols = [c for c in y_cols_for_attacks(attacks) if c in merged.columns]

    bundle_masks = None
    ds = data_dir / "pyg" / "pyg_dataset.pt"
    if ds.is_file():
        import torch

        try:
            bundle_masks = torch.load(ds, map_location="cpu", weights_only=False).get("split_masks")
        except TypeError:
            bundle_masks = torch.load(ds, map_location="cpu").get("split_masks")

    train_mask, test_mask = resolve_train_test_masks(meta, bundle_masks, split, manifest=manifest)
    tr_map = dict(zip(meta["window_id"].astype(str), train_mask))
    te_map = dict(zip(meta["window_id"].astype(str), test_mask))
    wids = merged["window_id"].astype(str)
    tr = np.array([tr_map[str(w)] for w in wids])
    te = np.array([te_map[str(w)] for w in wids])

    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])
    model = make_model("lightgbm", n_est, 8, 31, -1)
    model.fit(pd.DataFrame(X_tr, columns=x_cols), Y[tr])
    pred = model.predict(pd.DataFrame(X_te, columns=x_cols))

    from gnn.metrics import metrics_by_attack

    by_head = metrics_by_attack(Y[te], pred, attacks=attacks)
    report = {
        "model": "lightgbm",
        "split": split,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "test_r2": float(r2_score(Y[te].ravel(), pred.ravel())),
        "test_mae": float(mean_absolute_error(Y[te].ravel(), pred.ravel())),
        "attacks": list(attacks),
    }
    for head in attacks:
        report[head] = by_head[head]
        report[f"test_{head}_r2"] = float(by_head[head]["r2"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def eval_gnn_and_save(
    ckpt: Path,
    data_dir: Path,
    dataset: Path,
    global_csv: Path,
    split: str,
    out_json: Path,
    device: str,
    meta_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> dict[str, float]:
    if out_json.is_file():
        met = metrics_from_report_json(out_json)
        if met:
            return met
    met = eval_gnn_checkpoint(
        ckpt,
        dataset,
        data_dir,
        global_csv,
        device=device,
        split=split,
        meta_df=meta_df,
        manifest_df=manifest_df,
    )
    attacks = parse_attacks("no-capacity")
    report = {
        "split": f"{split_key_prefix(split)}_test",
        "n_test": None,
        "attacks": list(attacks),
        "overall": {"r2": met["overall_r2"], "mae": met["overall_mae"]},
        "betweenness": {"r2": met["betweenness_r2"]},
        "random": {"r2": met["random_r2"]},
        "checkpoint": str(ckpt),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return met


def log_model_to_mlflow(
    *,
    mlflow_enabled: bool,
    experiment: str,
    tracking_uri: str | None,
    split: str,
    label: str,
    model_slug: str,
    met: dict[str, float],
    artifact_path: Path,
    ckpt_path: Path | None,
    n_train: int,
    n_test: int,
    train_stratum: dict[str, float],
    test_stratum: dict[str, float],
    notes: str = "",
    skipped_disk: bool = False,
) -> None:
    if not mlflow_enabled:
        return
    run_name = f"{split}__{model_slug}"
    tracker = MlflowTracker(
        enabled=True,
        experiment_name=experiment,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags={
            "split": split,
            "model": label,
            "pipeline": "stratified_eval",
            "skipped_disk": str(skipped_disk),
        },
    )
    tracker.log_params(
        {
            "split": split,
            "model": label,
            "checkpoint": str(ckpt_path) if ckpt_path else "lgb_retrain",
            "n_train": n_train,
            "n_test": n_test,
            "notes": notes,
        }
    )
    log_stratum_metrics(tracker, train_stratum, "train")
    log_stratum_metrics(tracker, test_stratum, "test")
    metrics = {
        "test_overall_r2": float(met.get("overall_r2", np.nan)),
        "test_betweenness_r2": float(met.get("betweenness_r2", np.nan)),
        "test_random_r2": float(met.get("random_r2", np.nan)),
    }
    if "overall_mae" in met:
        metrics["test_overall_mae"] = float(met["overall_mae"])
    tracker.log_metrics(metrics)
    tracker.log_artifact(artifact_path)
    tracker.end()


def log_split_summary_mlflow(
    *,
    mlflow_enabled: bool,
    experiment: str,
    tracking_uri: str | None,
    split: str,
    split_dir: Path,
) -> None:
    if not mlflow_enabled:
        return
    tracker = MlflowTracker(
        enabled=True,
        experiment_name=experiment,
        run_name=f"{split}__summary",
        tracking_uri=tracking_uri,
        tags={"split": split, "pipeline": "stratified_eval", "aggregate": "true"},
    )
    for name in ("comparison_table.csv", "comparison_table.md"):
        p = split_dir / name
        if p.is_file():
            tracker.log_artifact(p, artifact_path=f"summary/{name}")
    tracker.end()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified-split eval for all manifest models")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--manifest", default="experiments/enhancement_manifest.json")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--global-csv", default="results_gnn/global_extended/window_global_features.csv")
    parser.add_argument("--out-root", default="results_gnn/stratified_eval")
    parser.add_argument("--splits", default="stratified_stratum,stratified_state")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", default=True)
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    parser.add_argument("--mlflow-experiment", default="ev-charging-stratified-splits")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="e.g. file:./mlruns or sqlite:///mlflow.db")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    out_root = Path(args.out_root).resolve()
    dataset = (data_dir / args.dataset).resolve()
    global_csv = data_dir / args.global_csv
    if not global_csv.is_file():
        global_csv = data_dir / "results_gnn/global_extended/window_global_features.csv"

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    manifest_data = json.loads((data_dir / args.manifest).read_text(encoding="utf-8"))
    experiments = manifest_data.get("experiments", manifest_data)

    meta_df = pd.read_csv(data_dir / "windows_meta.csv")
    manifest_df = pd.read_csv(data_dir / "manifest.csv")

    # Ensure masks exist in bundle (optional rebuild message)
    ds_path = dataset
    if ds_path.is_file():
        import torch

        try:
            bundle = torch.load(ds_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(ds_path, map_location="cpu")
        sm = bundle.get("split_masks", {})
        if "stratified_stratum_train" not in sm:
            print(
                "[warn] pyg_dataset.pt missing stratified masks; eval uses on-the-fly masks from meta/manifest.",
                flush=True,
            )

    all_tables: dict[str, pd.DataFrame] = {}

    for split in splits:
        print(f"\n{'='*60}\nSPLIT: {split}\n{'='*60}", flush=True)
        from gnn.splits import resolve_train_test_masks

        tr, te = resolve_train_test_masks(meta_df, None, split, manifest=manifest_df)
        print(format_stratum_report(meta_df, tr, te), flush=True)
        train_stratum = stratum_distribution(meta_df, tr)
        test_stratum = stratum_distribution(meta_df, te)
        n_train_split = int(tr.sum())
        n_test_split = int(te.sum())

        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        for entry in experiments:
            label = entry.get("label", "model")
            model_slug = slug(label)
            out_json = split_dir / model_slug / "test_metrics.json"

            if args.skip_existing and out_json.is_file():
                met = metrics_from_report_json(out_json)
                if met:
                    rows.append({"model": label, "notes": entry.get("notes", ""), **met, "source": str(out_json)})
                    print(f"  SKIP {label}", flush=True)
                    log_model_to_mlflow(
                        mlflow_enabled=args.mlflow,
                        experiment=args.mlflow_experiment,
                        tracking_uri=args.mlflow_tracking_uri,
                        split=split,
                        label=label,
                        model_slug=model_slug,
                        met=met,
                        artifact_path=out_json,
                        ckpt_path=Path(entry.get("ckpt", "")) if entry.get("ckpt") else None,
                        n_train=n_train_split,
                        n_test=n_test_split,
                        train_stratum=train_stratum,
                        test_stratum=test_stratum,
                        notes=entry.get("notes", ""),
                        skipped_disk=True,
                    )
                    continue

            ckpt = entry.get("ckpt") or entry.get("checkpoint")
            ckpt_path = None
            if ckpt:
                ckpt_path = Path(ckpt) if Path(ckpt).is_file() else data_dir / ckpt

            if ckpt_path is not None and ckpt_path.is_file():
                print(f"  GNN eval {label} <- {ckpt_path.name}", flush=True)
                met = eval_gnn_and_save(
                    ckpt_path,
                    data_dir,
                    dataset,
                    global_csv,
                    split,
                    out_json,
                    args.device,
                    meta_df,
                    manifest_df,
                )
                rows.append(
                    {
                        "model": label,
                        "notes": entry.get("notes", ""),
                        "betweenness_r2": met["betweenness_r2"],
                        "random_r2": met["random_r2"],
                        "overall_r2": met["overall_r2"],
                        "source": str(out_json),
                    }
                )
                log_model_to_mlflow(
                    mlflow_enabled=args.mlflow,
                    experiment=args.mlflow_experiment,
                    tracking_uri=args.mlflow_tracking_uri,
                    split=split,
                    label=label,
                    model_slug=model_slug,
                    met=met,
                    artifact_path=out_json,
                    ckpt_path=ckpt_path,
                    n_train=n_train_split,
                    n_test=n_test_split,
                    train_stratum=train_stratum,
                    test_stratum=test_stratum,
                    notes=entry.get("notes", ""),
                )
                continue

            if "lightgbm" in label.lower() or "lgb" in label.lower():
                lgb_out = data_dir / f"lgbm/train_lgbm_ext_br_{split}"
                if args.skip_existing and (lgb_out / "lgb_report.json").is_file():
                    rp = lgb_out / "lgb_report.json"
                else:
                    print(f"  LGB train+eval {label} -> {lgb_out}", flush=True)
                    rp = run_lgb_split(
                        data_dir,
                        split,
                        lgb_out,
                        global_csv,
                        parse_attacks("no-capacity"),
                        args.n_estimators,
                        skip_existing=args.skip_existing,
                    )
                met = metrics_from_report_json(rp)
                if met:
                    rows.append({"model": label, "notes": entry.get("notes", ""), **met, "source": str(rp)})
                    log_model_to_mlflow(
                        mlflow_enabled=args.mlflow,
                        experiment=args.mlflow_experiment,
                        tracking_uri=args.mlflow_tracking_uri,
                        split=split,
                        label=label,
                        model_slug=model_slug,
                        met=met,
                        artifact_path=rp,
                        ckpt_path=None,
                        n_train=n_train_split,
                        n_test=n_test_split,
                        train_stratum=train_stratum,
                        test_stratum=test_stratum,
                        notes=entry.get("notes", ""),
                        skipped_disk=args.skip_existing and rp.is_file(),
                    )
                continue

            print(f"  WARN skip {label}: no checkpoint found", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(split_dir / "comparison_table.csv", index=False)
        md = format_md(df)
        (split_dir / "comparison_table.md").write_text(md + "\n", encoding="utf-8")
        all_tables[split] = df
        print(f"\nSaved {split_dir / 'comparison_table.md'}", flush=True)
        log_split_summary_mlflow(
            mlflow_enabled=args.mlflow,
            experiment=args.mlflow_experiment,
            tracking_uri=args.mlflow_tracking_uri,
            split=split,
            split_dir=split_dir,
        )

    # Combined doc: existing east_west (from manifest) + new splits
    lines = [
        "# All models — split comparison",
        "",
        "East/West rows use **existing** eval artifacts (not re-run).",
        "Stratified rows are under `results_gnn/stratified_eval/`.",
        "",
    ]
    east_rows = []
    for entry in experiments:
        label = entry.get("label", "")
        ev = entry.get("eval_json") or entry.get("eval")
        if not ev:
            continue
        p = Path(ev) if Path(ev).is_file() else data_dir / ev
        met = metrics_from_report_json(p)
        if met:
            east_rows.append({"model": label, "split": "east_west", **met})
    if east_rows:
        lines.append("## East / West (existing results)")
        lines.append("")
        edf = pd.DataFrame(east_rows)
        lines.append(format_md(edf.rename(columns={"model": "model"})))
        lines.append("")

    for split, df in all_tables.items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append(format_md(df))
        lines.append("")

    combined_path = out_root / "all_splits_comparison.md"
    combined_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCombined report -> {combined_path}", flush=True)

    if args.mlflow:
        tracker = MlflowTracker(
            enabled=True,
            experiment_name=args.mlflow_experiment,
            run_name="all_splits_summary",
            tracking_uri=args.mlflow_tracking_uri,
            tags={"pipeline": "stratified_eval", "aggregate": "all"},
        )
        tracker.log_artifact(combined_path)
        tracker.log_artifact(out_root / "stratified_stratum/comparison_table.csv", artifact_path="stratified_stratum")
        tracker.log_artifact(out_root / "stratified_state/comparison_table.csv", artifact_path="stratified_state")
        tracker.end()
        uri = args.mlflow_tracking_uri or "file:./mlruns (default)"
        print(f"\n[MLflow] View UI: mlflow ui --backend-store-uri {uri}", flush=True)


if __name__ == "__main__":
    main()
