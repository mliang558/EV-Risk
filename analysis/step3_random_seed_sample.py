#!/usr/bin/env python3
"""
Step 3 (random seed): sample 10k windows by random charging-station seeds.

- No grid / no min center separation (same site may seed many windows).
- Three radii with quota 5:3:2 (10 / 30 / 80 km → 5000 / 3000 / 2000).
- Drop windows with < n_min nodes (default 15); cap at n_max (default 100).

Outputs (Step-4 compatible):
  windows_meta.csv
  windows_nodes.parquet
  sampling_report_random_seed.json

Usage:
  python step3_random_seed_sample.py \\
    --network-dir network_structures \\
    --out /opt/data_repo/mliang_work/step4_gpu \\
    --target-total 10000 \\
    --n-min 15
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except ImportError as e:
    raise ImportError("Requires scikit-learn (BallTree).") from e

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_charging_network import STATES_ORDER

EARTH_RADIUS_KM = 6371.0

# radius_km, window_stratum label (for downstream), default count at 10k total
RADIUS_TIERS = (
    {"radius_km": 10.0, "window_stratum": "urban", "fraction": 0.50},
    {"radius_km": 30.0, "window_stratum": "suburban", "fraction": 0.30},
    {"radius_km": 80.0, "window_stratum": "rural", "fraction": 0.20},
)


def load_national_nodes(network_dir: Path) -> pd.DataFrame:
    rows = []
    for state in STATES_ORDER:
        pkl_path = network_dir / f"network_{state}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        G = payload["network"] if isinstance(payload, dict) else payload
        for local_id, data in G.nodes(data=True):
            rows.append(
                {
                    "node_id": f"{state}_{local_id}",
                    "state": state,
                    "local_id": int(local_id),
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "capacity": float(data.get("capacity", 0.0)),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No network pickles in {network_dir}")
    return pd.DataFrame(rows)


def tier_targets(total: int, fractions: tuple[float, ...]) -> list[int]:
    raw = [int(round(total * f)) for f in fractions]
    diff = total - sum(raw)
    raw[0] += diff
    return raw


def sample_tier(
    nodes: pd.DataFrame,
    tree: BallTree,
    *,
    radius_km: float,
    window_stratum: str,
    target: int,
    rng: np.random.Generator,
    n_min: int,
    n_max: int,
    max_attempts_factor: int,
    assign_node_stratum: bool,
    tract_strata,
    project_root: Path,
    window_id_start: int = 0,
    initial_centers: list[tuple[float, float]] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    del initial_centers  # dedupe is post-hoc in run_random_seed_sampling
    radius_rad = radius_km / EARTH_RADIUS_KM
    n_pool = len(nodes)
    meta_rows: list[dict] = []
    node_rows: list[dict] = []
    attempts = 0
    rejected_small = 0
    max_attempts = max(target * max_attempts_factor, target * 50)

    while len(meta_rows) < target and attempts < max_attempts:
        attempts += 1
        seed_idx = int(rng.integers(0, n_pool))
        seed = nodes.iloc[seed_idx]
        lat, lon = float(seed["lat"]), float(seed["lon"])

        inds, _ = tree.query_radius(
            np.radians([[lat, lon]]),
            r=radius_rad,
            return_distance=True,
            sort_results=True,
        )
        global_inds = inds[0]
        if len(global_inds) < n_min:
            rejected_small += 1
            continue

        n_raw = len(global_inds)
        pick = (
            rng.choice(global_inds, size=n_max, replace=False)
            if n_raw > n_max
            else global_inds
        )
        window_nodes = nodes.iloc[pick]
        window_id = f"w_{window_stratum}_{window_id_start + len(meta_rows):05d}"

        stratum_mix: dict = {}
        if assign_node_stratum and "stratum" in window_nodes.columns:
            stratum_mix = window_nodes["stratum"].value_counts(normalize=True).to_dict()

        meta_rows.append(
            {
                "window_id": window_id,
                "window_stratum": window_stratum,
                "radius_km": radius_km,
                "sampling": "random_seed",
                "seed_node_id": seed["node_id"],
                "center_lat": lat,
                "center_lon": lon,
                "n_nodes_raw": int(n_raw),
                "n_nodes": int(len(pick)),
                "frac_urban": float(stratum_mix.get("urban", 0.0)),
                "frac_suburban": float(stratum_mix.get("suburban", 0.0)),
                "frac_rural": float(stratum_mix.get("rural", 0.0)),
            }
        )
        for _, nrow in window_nodes.iterrows():
            node_rows.append(
                {
                    "window_id": window_id,
                    "node_id": nrow["node_id"],
                    "state": nrow["state"],
                    "local_id": nrow["local_id"],
                    "lat": nrow["lat"],
                    "lon": nrow["lon"],
                    "capacity": nrow["capacity"],
                    "node_stratum": nrow.get("stratum", ""),
                    "tract_geoid": nrow.get("tract_geoid", ""),
                }
            )

    stats = {
        "window_stratum": window_stratum,
        "radius_km": radius_km,
        "target": target,
        "accepted": len(meta_rows),
        "attempts": attempts,
        "rejected_n_below_min": rejected_small,
        "success_rate": len(meta_rows) / max(attempts, 1),
    }
    return meta_rows, node_rows, stats


def _id_start_for_stratum(meta_rows: list[dict], stratum: str) -> int:
    return sum(1 for m in meta_rows if m.get("window_stratum") == stratum)


def _sample_round(
    nodes: pd.DataFrame,
    tree: BallTree,
    rng: np.random.Generator,
    counts: list[int],
    meta_rows: list[dict],
    *,
    n_min: int,
    n_max: int,
    max_attempts_factor: int,
    assign_node_stratum: bool,
    tract_strata,
    project_root: Path,
    dedupe_centers: bool,
    verbose: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Append one sampling round; return (new_meta, new_nodes, tier_stats)."""
    new_meta: list[dict] = []
    new_nodes: list[dict] = []
    tier_stats: list[dict] = []

    for tier, n_target in zip(RADIUS_TIERS, counts):
        if n_target <= 0:
            continue
        stratum = tier["window_stratum"]

        if verbose:
            print(
                f"\n=== {stratum} | R={tier['radius_km']} km | +{n_target} windows ===",
                flush=True,
            )
        meta, nrows, stats = sample_tier(
            nodes,
            tree,
            radius_km=float(tier["radius_km"]),
            window_stratum=stratum,
            target=int(n_target),
            rng=rng,
            n_min=n_min,
            n_max=n_max,
            max_attempts_factor=max_attempts_factor,
            assign_node_stratum=assign_node_stratum,
            tract_strata=tract_strata,
            project_root=project_root,
            window_id_start=_id_start_for_stratum(meta_rows + new_meta, stratum),
        )
        new_meta.extend(meta)
        new_nodes.extend(nrows)
        tier_stats.append(stats)
        if verbose:
            print(
                f"  Accepted {stats.get('accepted', len(meta))}/{n_target} | "
                f"attempts={stats.get('attempts', 0)}",
                flush=True,
            )
    return new_meta, new_nodes, tier_stats


def run_random_seed_sampling(
    network_dir: Path,
    out_dir: Path,
    target_total: int,
    n_min: int,
    n_max: int,
    seed: int,
    max_attempts_factor: int,
    geocode_method: str,
    tier_counts: dict[str, int] | None = None,
    oversample_factor: float = 1.0,
    dedupe_centers: bool = False,
    min_sep_frac: float = 0.2,
    dedupe_fill_rounds: int = 8,
    verbose: bool = True,
) -> dict:
    project_root = _ROOT.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading nodes from {network_dir}", flush=True)
    nodes = load_national_nodes(network_dir)
    if verbose:
        print(f"  National nodes: {len(nodes):,}", flush=True)

    assign_node_stratum = False
    try:
        from census_tract_strata import assign_nodes_to_tract_strata, load_or_build_tract_strata

        tract_strata = load_or_build_tract_strata(project_root)
        nodes = assign_nodes_to_tract_strata(
            nodes, tract_strata, project_root, geocode_method=geocode_method
        )
        assign_node_stratum = True
        if verbose:
            print("  Assigned tract strata to nodes (for frac_* in meta).", flush=True)
    except Exception as e:
        tract_strata = None
        if verbose:
            print(f"  [warn] tract strata skipped: {e}", flush=True)

    coords_rad = np.radians(nodes[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    rng = np.random.default_rng(seed)

    sample_total = target_total
    if tier_counts is None and oversample_factor and oversample_factor > 1.0:
        sample_total = int(round(target_total * oversample_factor))

    if tier_counts:
        base = dict(tier_counts)
        if sample_total != target_total:
            scale = sample_total / max(sum(base.values()), 1)
            counts = [max(1, int(round(base[t["window_stratum"]] * scale))) for t in RADIUS_TIERS]
            counts[0] += sample_total - sum(counts)
        else:
            counts = [int(tier_counts[t["window_stratum"]]) for t in RADIUS_TIERS]
        quota_note = f"custom: {dict(zip([t['window_stratum'] for t in RADIUS_TIERS], counts))}"
    else:
        fracs = tuple(t["fraction"] for t in RADIUS_TIERS)
        counts = tier_targets(sample_total, fracs)
        quota_note = "10km:30km:80km = 5:3:2 (default)"
    if sample_total != target_total:
        quota_note += f" | oversample x{oversample_factor} -> {sample_total} before dedupe"

    base_counts = list(counts)
    all_meta: list[dict] = []
    all_nodes: list[dict] = []
    report: dict = {
        "method": "random_seed",
        "target_total": target_total,
        "n_min": n_min,
        "n_max": n_max,
        "seed": seed,
        "quota": quota_note,
        "tier_counts_target": dict(zip([t["window_stratum"] for t in RADIUS_TIERS], base_counts)),
        "tiers": [],
        "dedupe_rounds": [],
    }

    from step3_dedupe_window_centers import greedy_dedupe, subsample_to_target

    if dedupe_centers:
        init_pool = int(target_total * max(oversample_factor, 3.0))
        init_counts = [
            max(1, int(round(init_pool * (base_counts[i] / max(sum(base_counts), 1)))))
            for i in range(len(RADIUS_TIERS))
        ]
        init_counts[0] += init_pool - sum(init_counts)
        if verbose:
            print(
                f"\nDedupe mode: initial pool ~{sum(init_counts):,} "
                f"(target {target_total:,}, min_sep=R*{min_sep_frac})",
                flush=True,
            )
    else:
        init_counts = base_counts

    nm, nn, tstats = _sample_round(
        nodes,
        tree,
        rng,
        init_counts,
        all_meta,
        n_min=n_min,
        n_max=n_max,
        max_attempts_factor=max_attempts_factor,
        assign_node_stratum=assign_node_stratum,
        tract_strata=tract_strata,
        project_root=project_root,
        dedupe_centers=dedupe_centers,
        verbose=verbose,
    )
    all_meta.extend(nm)
    all_nodes.extend(nn)
    report["tiers"].extend(tstats)

    meta_df = pd.DataFrame(all_meta)
    kept_df = meta_df

    if dedupe_centers:
        for round_i in range(1, dedupe_fill_rounds + 1):
            kept_df, dstats = greedy_dedupe(
                meta_df, rng, min_sep_frac=min_sep_frac, within_stratum_only=True
            )
            retention = len(kept_df) / max(len(meta_df), 1)
            report["dedupe_rounds"].append(
                {"round": round_i, "pool": len(meta_df), "kept": len(kept_df), "retention": retention}
            )
            if verbose:
                print(
                    f"\nDedupe round {round_i}: kept {len(kept_df):,} / {len(meta_df):,} "
                    f"(retention={retention:.1%})",
                    flush=True,
                )
            if len(kept_df) >= target_total:
                meta_df = subsample_to_target(kept_df, target_total, rng)
                break
            if round_i >= dedupe_fill_rounds:
                print(
                    f"[warn] After {dedupe_fill_rounds} rounds: {len(kept_df)} < {target_total}. "
                    "Try --min-sep-frac 0.15 or --oversample-factor 4",
                    flush=True,
                )
                meta_df = kept_df
                break

            shortfall = target_total - len(kept_df)
            add_total = int(shortfall / max(retention, 0.12) * 1.25) + 800
            add_counts = [
                max(80, int(add_total * (base_counts[i] / max(sum(base_counts), 1))))
                for i in range(len(RADIUS_TIERS))
            ]
            add_counts[0] += add_total - sum(add_counts)
            if verbose:
                print(
                    f"--- Fill round {round_i + 1}: need +{shortfall}, draw +{add_total} more ---",
                    flush=True,
                )
            nm, nn, tstats = _sample_round(
                nodes,
                tree,
                rng,
                add_counts,
                all_meta,
                n_min=n_min,
                n_max=n_max,
                max_attempts_factor=max_attempts_factor,
                assign_node_stratum=assign_node_stratum,
                tract_strata=tract_strata,
                project_root=project_root,
                dedupe_centers=True,
                verbose=verbose,
            )
            all_meta.extend(nm)
            all_nodes.extend(nn)
            report["tiers"].extend(tstats)
            meta_df = pd.DataFrame(all_meta)
        else:
            meta_df = kept_df
    elif len(meta_df) > target_total:
        meta_df = subsample_to_target(meta_df, target_total, rng)

    nodes_df = pd.DataFrame(all_nodes)
    report["sampled_before_dedupe"] = len(pd.DataFrame(all_meta))
    if len(meta_df) < target_total:
        report["dedupe_shortfall"] = target_total - len(meta_df)

    keep_ids = set(meta_df["window_id"].astype(str))
    nodes_df = nodes_df[nodes_df["window_id"].astype(str).isin(keep_ids)].copy()

    meta_df.to_csv(out_dir / "windows_meta.csv", index=False)
    nodes_df.to_parquet(out_dir / "windows_nodes.parquet", index=False)

    report["total_windows"] = len(meta_df)
    report["target_total"] = target_total
    report["by_stratum_summary"] = []
    for g, sub in meta_df.groupby("window_stratum"):
        report["by_stratum_summary"].append(
            {
                "window_stratum": str(g),
                "radius_km": float(sub["radius_km"].iloc[0]),
                "n_windows": int(len(sub)),
                "n_nodes_mean": float(sub["n_nodes"].mean()),
                "n_nodes_min": int(sub["n_nodes"].min()),
                "n_nodes_max": int(sub["n_nodes"].max()),
            }
        )

    with open(out_dir / "sampling_report_random_seed.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\nWrote {out_dir / 'windows_meta.csv'} ({len(meta_df):,} windows)")
        print(meta_df.groupby(["window_stratum", "radius_km"]).size())
        print("\nNext:")
        print("  python step4_export_window_subgraphs.py --windows-dir . --out .")
        print("  python step4_compute_y_labels.py --data-dir . --workers 16")
        print("  python step5_build_pyg_dataset.py --data-dir .")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-seed window sampling (10k, 5:3:2 radii)")
    parser.add_argument("--network-dir", default=None, help="Dir with network_XX.pkl (auto: ./network_structures)")
    parser.add_argument("--out", default=".")
    parser.add_argument("--target-total", type=int, default=10000, help="Final desired count (before dedupe)")
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=3.0,
        help="Initial pool = target * factor before dedupe (default 3.0; ~50-70%% retention at R*0.2)",
    )
    parser.add_argument(
        "--dedupe-fill-rounds",
        type=int,
        default=8,
        help="Max extra sampling rounds until dedupe pool reaches target",
    )
    parser.add_argument(
        "--dedupe-centers",
        action="store_true",
        help="After sampling, drop centers closer than R*min-sep-frac (default 0.2)",
    )
    parser.add_argument(
        "--min-sep-frac",
        type=float,
        default=0.2,
        help="Center min separation = radius * this (default 0.2 => 2 km at 10 km urban)",
    )
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts-factor", type=int, default=300)
    parser.add_argument("--geocode", choices=("fcc", "county"), default="fcc")
    parser.add_argument(
        "--probe-json",
        default=None,
        help="Use suggested_quota_by_success_rate from step3_probe_random_seed.py output",
    )
    parser.add_argument("--urban-count", type=int, default=None)
    parser.add_argument("--suburban-count", type=int, default=None)
    parser.add_argument("--rural-count", type=int, default=None)
    args = parser.parse_args()

    from network_paths import resolve_network_dir

    try:
        network_dir = resolve_network_dir(args.network_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    tier_counts = None
    if args.probe_json:
        with open(args.probe_json, encoding="utf-8") as f:
            probe = json.load(f)
        sq = probe["suggested_quota_by_success_rate"]
        tier_counts = {k: int(sq[k]["suggested_n_windows"]) for k in ("urban", "suburban", "rural")}
        print(f"Quotas from probe: {tier_counts}", flush=True)
    elif any(x is not None for x in (args.urban_count, args.suburban_count, args.rural_count)):
        tier_counts = {
            "urban": args.urban_count if args.urban_count is not None else 5000,
            "suburban": args.suburban_count if args.suburban_count is not None else 3000,
            "rural": args.rural_count if args.rural_count is not None else 2000,
        }
        print(f"Quotas manual: {tier_counts}", flush=True)

    run_random_seed_sampling(
        network_dir=network_dir,
        out_dir=Path(args.out).resolve(),
        target_total=args.target_total,
        n_min=args.n_min,
        n_max=args.n_max,
        seed=args.seed,
        max_attempts_factor=args.max_attempts_factor,
        geocode_method=args.geocode,
        tier_counts=tier_counts,
        oversample_factor=args.oversample_factor if args.dedupe_centers else 1.0,
        dedupe_centers=args.dedupe_centers,
        min_sep_frac=args.min_sep_frac,
        dedupe_fill_rounds=args.dedupe_fill_rounds,
    )


if __name__ == "__main__":
    main()
