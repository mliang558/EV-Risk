#!/usr/bin/env python3
"""
Post-filter sampled windows: drop centers closer than R*min_sep_frac (greedy, random order).

Typical pipeline:
  1) Oversample: step3_random_seed_sample.py --target-total 15000 ...
  2) Dedupe:    step3_dedupe_window_centers.py --target 10000
  3) If still < target, sample more and run dedupe again.

Usage:
  python step3_dedupe_window_centers.py \\
    --meta windows_meta.csv \\
    --nodes windows_nodes.parquet \\
    --out windows_meta_deduped.csv \\
    --nodes-out windows_nodes_deduped.parquet \\
    --target 10000 \\
    --min-sep-frac 0.2 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except ImportError as e:
    raise ImportError("Requires scikit-learn.") from e

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def min_sep_km(radius_km: float, frac: float) -> float:
    return float(radius_km) * float(frac)


def greedy_dedupe(
    meta: pd.DataFrame,
    rng: np.random.Generator,
    min_sep_frac: float = 0.2,
    within_stratum_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Random order greedy: keep window if no kept window has center within max(ri,rj)*frac.
    """
    meta = meta.reset_index(drop=True)
    order = rng.permutation(len(meta))

    kept_mask = np.zeros(len(meta), dtype=bool)
    kept_lats: list[float] = []
    kept_lons: list[float] = []
    kept_radii: list[float] = []
    kept_strata: list[str] = []
    dropped_pairs = 0

    for idx in order:
        row = meta.iloc[idx]
        lat = float(row["center_lat"])
        lon = float(row["center_lon"])
        r = float(row["radius_km"])
        sep_need = min_sep_km(r, min_sep_frac)
        stratum = str(row.get("window_stratum", ""))

        conflict = False
        if kept_lats:
            lats = np.array(kept_lats)
            lons = np.array(kept_lons)
            dists = haversine_km(lat, lon, lats, lons)

            if within_stratum_only:
                same = np.array([s == stratum for s in kept_strata])
                if same.any():
                    for d, rj, s_ok in zip(dists, kept_radii, same):
                        if not s_ok:
                            continue
                        thresh = max(sep_need, min_sep_km(rj, min_sep_frac))
                        if d < thresh:
                            conflict = True
                            dropped_pairs += 1
                            break
            else:
                thresh = np.maximum(sep_need, np.array([min_sep_km(rj, min_sep_frac) for rj in kept_radii]))
                if np.any(dists < thresh):
                    conflict = True
                    dropped_pairs += 1

        if conflict:
            continue

        kept_mask[idx] = True
        kept_lats.append(lat)
        kept_lons.append(lon)
        kept_radii.append(r)
        kept_strata.append(stratum)

    out = meta.loc[kept_mask].copy()
    stats = {
        "input_windows": int(len(meta)),
        "kept_after_dedupe": int(len(out)),
        "dropped": int(len(meta) - len(out)),
        "min_sep_frac": min_sep_frac,
        "within_stratum_only": within_stratum_only,
        "drop_rate": float(1.0 - len(out) / max(len(meta), 1)),
    }
    return out, stats


def subsample_to_target(meta: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(meta) <= target:
        return meta
    idx = rng.choice(len(meta), size=target, replace=False)
    return meta.iloc[idx].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe window centers (distance >= R*min_sep_frac)")
    parser.add_argument("--meta", default="windows_meta.csv")
    parser.add_argument("--nodes", default="windows_nodes.parquet")
    parser.add_argument("--out", default="windows_meta.csv", help="Filtered meta (overwrite ok)")
    parser.add_argument("--nodes-out", default="windows_nodes.parquet")
    parser.add_argument("--target", type=int, default=10000, help="Final window count cap")
    parser.add_argument(
        "--min-sep-frac",
        type=float,
        default=0.2,
        help="Minimum center separation = radius * this (default 0.2)",
    )
    parser.add_argument(
        "--cross-stratum",
        action="store_true",
        help="Also dedupe across radii (default: only same window_stratum)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="results_sampling/dedupe_report.json")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    nodes_path = Path(args.nodes).resolve()
    meta = pd.read_csv(meta_path)
    nodes = pd.read_parquet(nodes_path)
    rng = np.random.default_rng(args.seed)

    print(f"Input: {len(meta):,} windows", flush=True)
    deduped, stats = greedy_dedupe(
        meta,
        rng,
        min_sep_frac=args.min_sep_frac,
        within_stratum_only=not args.cross_stratum,
    )
    print(
        f"After dedupe (min_sep=R*{args.min_sep_frac}): {len(deduped):,} "
        f"(dropped {stats['dropped']:,})",
        flush=True,
    )

    if len(deduped) > args.target:
        deduped = subsample_to_target(deduped, args.target, rng)
        stats["subsampled_to_target"] = args.target
        print(f"Subsampled to target: {len(deduped):,}", flush=True)
    elif len(deduped) < args.target:
        stats["shortfall"] = args.target - len(deduped)
        print(
            f"[warn] Only {len(deduped):,} windows after dedupe (target {args.target:,}). "
            "Oversample more then re-run dedupe.",
            flush=True,
        )

    keep_ids = set(deduped["window_id"].astype(str))
    nodes_out = nodes[nodes["window_id"].astype(str).isin(keep_ids)].copy()

    out_meta = Path(args.out).resolve()
    out_nodes = Path(args.nodes_out).resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(out_meta, index=False)
    nodes_out.to_parquet(out_nodes, index=False)

    stats["target"] = args.target
    stats["final_windows"] = int(len(deduped))
    stats["by_stratum"] = (
        deduped.groupby("window_stratum").size().astype(int).to_dict() if len(deduped) else {}
    )

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nBy stratum:")
    if len(deduped):
        print(deduped.groupby(["window_stratum", "radius_km"]).size())
    print(f"\nWrote {out_meta}")
    print(f"Wrote {out_nodes}")
    print(f"Report {report_path}")


if __name__ == "__main__":
    main()
