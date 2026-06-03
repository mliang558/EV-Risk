#!/usr/bin/env python3
"""
Post-filter sampled windows: enforce max disk overlap (default 50%%) between kept windows.

Greedy keep (default placement: small radius first): drop candidate if overlap exceeds threshold.
By default checks **all strata** (urban / suburban / rural do not overlap each other).

Typical pipeline:
  1) Oversample: step3_random_seed_sample.py --dedupe-centers --max-overlap-frac 0.5 ...
  2) Or:       step3_dedupe_window_centers.py --max-overlap-frac 0.5 --cross-stratum

Usage:
  python step3_dedupe_window_centers.py \\
    --meta windows_meta.csv --nodes windows_nodes.parquet \\
    --target 10000 --max-overlap-frac 0.5 --cross-stratum --seed 42
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

from gnn.window_overlap import (
    disk_overlap_fraction,
    haversine_km,
    sep_frac_for_max_overlap,
    windows_conflict,
)

EARTH_RADIUS_KM = 6371.0


def _placement_order(meta: pd.DataFrame, rng: np.random.Generator, mode: str) -> np.ndarray:
    """radius_asc: place 10 km urban before 30/80 km (fits many more under cross-stratum rules)."""
    n = len(meta)
    if mode == "random":
        return rng.permutation(n)
    r = meta["radius_km"].astype(float).values
    jitter = rng.random(n) * 1e-3
    if mode == "radius_desc":
        return np.argsort(-r + jitter)
    return np.argsort(r + jitter)


def greedy_dedupe(
    meta: pd.DataFrame,
    rng: np.random.Generator,
    *,
    max_overlap_frac: float = 0.5,
    within_stratum_only: bool = False,
    cross_stratum_mode: str = "area",
    placement_order: str = "radius_asc",
    min_sep_frac: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Greedy keep: no conflict with already-kept windows.

    Use placement_order=radius_asc (default) when deduping across strata so small
    urban disks are placed before large rural disks (~10k feasible). Random order
    with strict cross-stratum area overlap often keeps only ~1–2k from 50k pool.
    """
    meta = meta.reset_index(drop=True)
    if within_stratum_only:
        order = _placement_order(meta, rng, placement_order)
    else:
        order = _placement_order(meta, rng, placement_order if placement_order != "random" else "radius_asc")

    kept_mask = np.zeros(len(meta), dtype=bool)
    kept_lats: list[float] = []
    kept_lons: list[float] = []
    kept_radii: list[float] = []
    kept_strata: list[str] = []
    overlap_rejects = 0
    sep_rejects = 0

    for idx in order:
        row = meta.iloc[idx]
        lat = float(row["center_lat"])
        lon = float(row["center_lon"])
        r = float(row["radius_km"])
        stratum = str(row.get("window_stratum", ""))

        conflict = False
        for kl, klo, kr, ks in zip(kept_lats, kept_lons, kept_radii, kept_strata):
            if within_stratum_only and ks != stratum:
                continue
            mode = "area" if (within_stratum_only or ks == stratum) else cross_stratum_mode
            if windows_conflict(
                lat, lon, r, stratum, kl, klo, kr, ks,
                max_overlap_frac=max_overlap_frac,
                cross_stratum_mode=mode,
            ):
                conflict = True
                overlap_rejects += 1
                break
            if min_sep_frac is not None:
                d = haversine_km(lat, lon, kl, klo)
                if d < max(r, kr) * float(min_sep_frac):
                    conflict = True
                    sep_rejects += 1
                    break

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
        "max_overlap_frac": max_overlap_frac,
        "within_stratum_only": within_stratum_only,
        "cross_stratum_mode": cross_stratum_mode,
        "placement_order": placement_order,
        "min_sep_frac": min_sep_frac,
        "overlap_rejects": overlap_rejects,
        "sep_rejects": sep_rejects,
        "drop_rate": float(1.0 - len(out) / max(len(meta), 1)),
        "equiv_sep_frac_urban_10km": sep_frac_for_max_overlap(max_overlap_frac, 10.0),
        "equiv_sep_frac_suburban_30km": sep_frac_for_max_overlap(max_overlap_frac, 30.0),
        "equiv_sep_frac_rural_80km": sep_frac_for_max_overlap(max_overlap_frac, 80.0),
    }
    return out, stats


def quota_dedupe_per_stratum(
    meta: pd.DataFrame,
    rng: np.random.Generator,
    *,
    tier_targets: dict[str, int],
    max_overlap_frac: float = 0.5,
    placement_order: str = "radius_asc",
    min_sep_frac: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Dedupe within each stratum (50%% area), up to per-tier quotas.

    Cross-stratum overlap is allowed; use this when global cross-stratum dedupe
  plateaus around ~2–3k on dense charging networks but ~10k within-stratum windows are needed.
    """
    parts: list[pd.DataFrame] = []
    per_stratum: dict[str, dict] = {}
    for stratum, cap in tier_targets.items():
        sub = meta[meta["window_stratum"].astype(str) == str(stratum)]
        if sub.empty or cap <= 0:
            per_stratum[stratum] = {"pool": 0, "kept": 0, "cap": cap}
            continue
        kept, st = greedy_dedupe(
            sub,
            rng,
            max_overlap_frac=max_overlap_frac,
            within_stratum_only=True,
            placement_order=placement_order,
            min_sep_frac=min_sep_frac,
        )
        if len(kept) > cap:
            kept = subsample_to_target(kept, cap, rng)
        parts.append(kept)
        per_stratum[stratum] = {"pool": int(len(sub)), "kept": int(len(kept)), "cap": cap, **st}

    out = pd.concat(parts, ignore_index=True) if parts else meta.iloc[0:0].copy()
    stats = {
        "dedupe_strategy": "per_stratum",
        "input_windows": int(len(meta)),
        "kept_after_dedupe": int(len(out)),
        "max_overlap_frac": max_overlap_frac,
        "placement_order": placement_order,
        "per_stratum": per_stratum,
        "by_stratum": out.groupby("window_stratum").size().astype(int).to_dict() if len(out) else {},
    }
    return out, stats


def subsample_to_target(meta: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(meta) <= target:
        return meta
    idx = rng.choice(len(meta), size=target, replace=False)
    return meta.iloc[idx].copy()


def verify_max_overlap(meta: pd.DataFrame, max_overlap_frac: float, within_stratum_only: bool) -> dict:
    """Pairwise scan (slow; for reporting only)."""
    max_seen = 0.0
    n_pairs = 0
    viol = 0
    rows = meta.reset_index(drop=True)
    for i in range(len(rows)):
        ri = rows.iloc[i]
        for j in range(i + 1, len(rows)):
            if within_stratum_only and ri["window_stratum"] != rows.iloc[j]["window_stratum"]:
                continue
            d = haversine_km(
                float(ri["center_lat"]),
                float(ri["center_lon"]),
                float(rows.iloc[j]["center_lat"]),
                float(rows.iloc[j]["center_lon"]),
            )
            ov = disk_overlap_fraction(float(ri["radius_km"]), float(rows.iloc[j]["radius_km"]), d)
            max_seen = max(max_seen, ov)
            n_pairs += 1
            if ov > max_overlap_frac + 1e-6:
                viol += 1
    return {"max_pairwise_overlap": max_seen, "n_pairs_checked": n_pairs, "violations": viol}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe windows by max disk overlap")
    parser.add_argument("--meta", default="windows_meta.csv")
    parser.add_argument("--nodes", default="windows_nodes.parquet")
    parser.add_argument("--out", default="windows_meta.csv")
    parser.add_argument("--nodes-out", default="windows_nodes.parquet")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument(
        "--max-overlap-frac",
        type=float,
        default=0.5,
        help="Max allowed overlap fraction of smaller disk (default 0.5 = 50%%)",
    )
    parser.add_argument(
        "--min-sep-frac",
        type=float,
        default=None,
        help="Optional extra center separation = R * frac (legacy; usually omit)",
    )
    parser.add_argument(
        "--within-stratum-only",
        action="store_true",
        help="Only dedupe within same stratum (default: dedupe across all strata)",
    )
    parser.add_argument(
        "--cross-stratum-mode",
        choices=("area", "center"),
        default="area",
        help="Cross-stratum conflict: area=50%% on smaller disk (strict); center=only if small center inside large disk",
    )
    parser.add_argument(
        "--placement-order",
        choices=("radius_asc", "radius_desc", "random"),
        default="radius_asc",
        help="Processing order (radius_asc strongly recommended for ~10k target)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="results_sampling/dedupe_report.json")
    parser.add_argument("--verify", action="store_true", help="Pairwise overlap audit on output")
    args = parser.parse_args()

    meta = pd.read_csv(Path(args.meta).resolve())
    nodes = pd.read_parquet(Path(args.nodes).resolve())
    rng = np.random.default_rng(args.seed)
    cross = not args.within_stratum_only

    print(f"Input: {len(meta):,} windows", flush=True)
    print(
        f"Dedupe: max_overlap={args.max_overlap_frac:.0%} | "
        f"cross_stratum={'yes' if cross else 'no (within stratum only)'}",
        flush=True,
    )
    deduped, stats = greedy_dedupe(
        meta,
        rng,
        max_overlap_frac=args.max_overlap_frac,
        within_stratum_only=not cross,
        cross_stratum_mode=args.cross_stratum_mode,
        placement_order=args.placement_order,
        min_sep_frac=args.min_sep_frac,
    )
    print(
        f"Kept {len(deduped):,} (dropped {stats['dropped']:,}, "
        f"overlap rejects {stats['overlap_rejects']:,})",
        flush=True,
    )

    if len(deduped) > args.target:
        deduped = subsample_to_target(deduped, args.target, rng)
        stats["subsampled_to_target"] = args.target
    elif len(deduped) < args.target:
        stats["shortfall"] = args.target - len(deduped)
        print(f"[warn] Shortfall {stats['shortfall']:,} vs target {args.target:,}", flush=True)

    if args.verify:
        v = verify_max_overlap(deduped, args.max_overlap_frac, within_stratum_only=not cross)
        stats["verify"] = v
        print(f"Verify: max pairwise overlap={v['max_pairwise_overlap']:.3f}, violations={v['violations']}", flush=True)

    keep_ids = set(deduped["window_id"].astype(str))
    nodes_out = nodes[nodes["window_id"].astype(str).isin(keep_ids)].copy()

    out_meta = Path(args.out).resolve()
    out_nodes = Path(args.nodes_out).resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(out_meta, index=False)
    nodes_out.to_parquet(out_nodes, index=False)

    stats["target"] = args.target
    stats["final_windows"] = int(len(deduped))
    stats["by_stratum"] = deduped.groupby("window_stratum").size().astype(int).to_dict() if len(deduped) else {}

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nBy stratum:")
    if len(deduped):
        print(deduped.groupby(["window_stratum", "radius_km"]).size())
    print(f"\nWrote {out_meta}\nWrote {out_nodes}\nReport {report_path}")


if __name__ == "__main__":
    main()
