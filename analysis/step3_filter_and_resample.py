#!/usr/bin/env python3
"""
Filter Step-3 windows (n_nodes >= 15) and resample each stratum to ~10k total.

Targets (default): urban 3334 @ 10km | suburban 3333 @ 30km | rural 3333 @ 80km

Then re-run on server:
  python step4_export_window_subgraphs.py --windows-dir .
  python step4_compute_y_labels.py --data-dir . --workers 16
  python step5_build_pyg_dataset.py --data-dir .
  python step6_train_gnn.py ...

Usage:
  python step3_filter_and_resample.py \\
    --existing-meta windows_meta.csv \\
    --existing-nodes windows_nodes.parquet \\
    --network-dir /path/to/network_structures \\
    --out . \\
    --n-min 15 \\
    --target-total 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from census_tract_strata import assign_nodes_to_tract_strata
from sample_sliding_windows_step3 import (
    STRATUM_CONFIG,
    STRATUM_TARGETS,
    load_national_nodes,
    load_or_build_tract_strata,
    run_sampling,
    sample_stratum,
)


def filter_meta(meta: pd.DataFrame, n_min: int) -> pd.DataFrame:
    m = meta[meta["n_nodes"] >= n_min].copy()
    return m.reset_index(drop=True)


def centers_from_meta(meta: pd.DataFrame) -> list[tuple[float, float]]:
    return list(zip(meta["center_lat"].astype(float), meta["center_lon"].astype(float)))


def resample_to_targets(
    network_dir: Path,
    kept_meta: pd.DataFrame,
    kept_nodes: pd.DataFrame,
    stratum_targets: dict[str, int],
    n_min: int,
    n_max: int,
    seed: int,
    geocode_method: str,
    max_attempts_factor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    project_root = _ROOT.parent
    nodes = load_national_nodes(network_dir)
    tract_strata = load_or_build_tract_strata(project_root)
    nodes = assign_nodes_to_tract_strata(nodes, tract_strata, project_root, geocode_method=geocode_method)
    rng = np.random.default_rng(seed)

    new_meta_rows: list[dict] = []
    new_node_rows: list[dict] = []
    report: dict = {"strata": {}, "dropped_below_n_min": 0}

    for stratum, target_n in stratum_targets.items():
        cfg = STRATUM_CONFIG[stratum]
        radius_km = float(cfg["radius_km"])
        sub_kept = kept_meta[kept_meta["window_stratum"] == stratum]
        have = len(sub_kept)
        need = max(0, int(target_n) - have)
        init_centers = centers_from_meta(sub_kept)
        id_start = have

        report["strata"][stratum] = {
            "radius_km": radius_km,
            "target": target_n,
            "kept": have,
            "to_sample": need,
        }

        if need <= 0:
            print(f"  {stratum}: kept {have} >= target {target_n}, skip sampling", flush=True)
            continue

        print(
            f"  {stratum}: R={radius_km}km | kept={have} | sampling +{need} (n_min={n_min})",
            flush=True,
        )
        meta_add, nodes_add, stats = sample_stratum(
            nodes,
            stratum=stratum,
            radius_km=radius_km,
            target=need,
            rng=rng,
            n_min=n_min,
            n_max=n_max,
            max_attempts_factor=max_attempts_factor,
            initial_centers=init_centers,
            window_id_start=id_start,
        )
        new_meta_rows.extend(meta_add)
        new_node_rows.extend(nodes_add)
        report["strata"][stratum]["sampled"] = stats.get("accepted", len(meta_add))
        report["strata"][stratum]["attempts"] = stats.get("attempts", 0)
        print(
            f"    -> got {len(meta_add)}/{need} new windows ({stats.get('attempts', 0)} attempts)",
            flush=True,
        )

    if new_meta_rows:
        add_meta = pd.DataFrame(new_meta_rows)
        add_nodes = pd.DataFrame(new_node_rows)
    else:
        add_meta = pd.DataFrame(columns=kept_meta.columns)
        add_nodes = pd.DataFrame(columns=kept_nodes.columns)

    out_meta = pd.concat([kept_meta, add_meta], ignore_index=True)
    out_nodes = pd.concat([kept_nodes, add_nodes], ignore_index=True)
    report["total_windows"] = len(out_meta)
    return out_meta, out_nodes, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter n>=15 and resample to 10k windows")
    parser.add_argument("--existing-meta", default="windows_meta.csv")
    parser.add_argument("--existing-nodes", default="windows_nodes.parquet")
    parser.add_argument("--network-dir", required=True)
    parser.add_argument("--out", default=".", help="Write windows_meta.csv + windows_nodes.parquet")
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geocode", choices=("fcc", "county"), default="fcc")
    parser.add_argument("--max-attempts-factor", type=int, default=150)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing meta; full resample from scratch (n_min=15)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    network_dir = Path(args.network_dir).resolve()
    if not network_dir.exists():
        print(f"Network dir not found: {network_dir}", file=sys.stderr)
        sys.exit(1)

    # Per-stratum targets summing to target_total
    base = dict(STRATUM_TARGETS)
    scale = args.target_total / sum(base.values())
    targets = {k: int(round(v * scale)) for k, v in base.items()}
    diff = args.target_total - sum(targets.values())
    targets["urban"] += diff

    if args.fresh:
        cfg = {k: {**v, "target": targets[k]} for k, v in STRATUM_CONFIG.items()}
        run_sampling(
            network_dir=network_dir,
            out_dir=out_dir,
            stratum_config=cfg,
            seed=args.seed,
            n_min=args.n_min,
            n_max=args.n_max,
            max_attempts_factor=args.max_attempts_factor,
            geocode_method=args.geocode,
        )
        return

    meta_path = Path(args.existing_meta)
    if not meta_path.is_absolute():
        meta_path = out_dir / meta_path
    nodes_path = Path(args.existing_nodes)
    if not nodes_path.is_absolute():
        nodes_path = out_dir / nodes_path

    meta = pd.read_csv(meta_path)
    nodes_df = pd.read_parquet(nodes_path)
    n_before = len(meta)
    kept_meta = filter_meta(meta, args.n_min)
    dropped_ids = set(meta.loc[meta["n_nodes"] < args.n_min, "window_id"])
    kept_nodes = nodes_df[~nodes_df["window_id"].isin(dropped_ids)].copy()

    print(f"Filtered n_nodes < {args.n_min}: {n_before} -> {len(kept_meta)} kept", flush=True)
    print(f"Targets: {targets} (total={sum(targets.values())})", flush=True)

    out_meta, out_nodes, report = resample_to_targets(
        network_dir,
        kept_meta,
        kept_nodes,
        targets,
        args.n_min,
        args.n_max,
        args.seed,
        args.geocode,
        args.max_attempts_factor,
    )

    report["n_min"] = args.n_min
    report["dropped_below_n_min"] = n_before - len(kept_meta)
    report["targets"] = targets

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = out_dir / "windows_meta.csv"
    nodes_out = out_dir / "windows_nodes.parquet"
    out_meta.to_csv(meta_out, index=False)
    out_nodes.to_parquet(nodes_out, index=False)
    with open(out_dir / "sampling_report_resample.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {meta_out} ({len(out_meta):,} windows)")
    print(out_meta.groupby("window_stratum")["n_nodes"].agg(["count", "mean", "min", "max"]))
    print("\nNext: step4_export -> step4 labels -> step5 pyg -> step6 train")


if __name__ == "__main__":
    main()
