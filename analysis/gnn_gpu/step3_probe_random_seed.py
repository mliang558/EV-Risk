#!/usr/bin/env python3
"""
Exploratory probe: per radius, try N random seeds and measure valid-window rate (n_nodes >= n_min).

Use this to set 10k sampling quotas (inverse to success rate or proportional to valid yield).

Usage (on server, from step4_gpu/):
  cd /opt/data_repo/mliang_work/step4_gpu
  ls network_structures/network_CA.pkl   # must exist
  python step3_probe_random_seed.py \\
    --network-dir network_structures \\
    --seeds-per-radius 100 --n-min 15

No windows_meta written — fast diagnostic only.
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
    raise ImportError("Requires scikit-learn.") from e

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from network_paths import resolve_network_dir
from run_charging_network import STATES_ORDER

EARTH_RADIUS_KM = 6371.0

RADII_KM = (10.0, 30.0, 80.0)
STRATUM_LABELS = ("urban", "suburban", "rural")


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
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No pickles in {network_dir}")
    return pd.DataFrame(rows)


def probe_radius(
    nodes: pd.DataFrame,
    tree: BallTree,
    radius_km: float,
    n_seeds: int,
    rng: np.random.Generator,
    n_min: int,
    n_max: int,
) -> dict:
    radius_rad = radius_km / EARTH_RADIUS_KM
    n_pool = len(nodes)
    n_nodes_list: list[int] = []
    valid = 0

    for _ in range(n_seeds):
        seed_idx = int(rng.integers(0, n_pool))
        row = nodes.iloc[seed_idx]
        lat, lon = float(row["lat"]), float(row["lon"])
        inds, _ = tree.query_radius(
            np.radians([[lat, lon]]),
            r=radius_rad,
            return_distance=True,
        )
        n_raw = int(len(inds[0]))
        n_nodes_list.append(n_raw)
        if n_min <= n_raw:
            valid += 1

    arr = np.array(n_nodes_list, dtype=np.int32)
    valid_mask = arr >= n_min
    valid_arr = arr[valid_mask]

    return {
        "radius_km": radius_km,
        "n_seeds_tried": n_seeds,
        "n_valid": int(valid),
        "success_rate": float(valid / n_seeds),
        "n_min": n_min,
        "n_max_cap": n_max,
        "n_nodes_all_mean": float(arr.mean()),
        "n_nodes_all_median": float(np.median(arr)),
        "n_nodes_all_p10": float(np.percentile(arr, 10)),
        "n_nodes_all_p90": float(np.percentile(arr, 90)),
        "n_nodes_valid_mean": float(valid_arr.mean()) if len(valid_arr) else None,
        "n_nodes_valid_median": float(np.median(valid_arr)) if len(valid_arr) else None,
        "pct_below_n_min": float((arr < n_min).mean()),
        "histogram": {
            "bins": [0, 5, 10, 15, 20, 30, 50, 100, 200, 10000],
            "counts": np.histogram(arr, bins=[0, 5, 10, 15, 20, 30, 50, 100, 200, 10000])[0].tolist(),
        },
    }


def suggest_quotas(results: list[dict], target_total: int) -> dict:
    """
    Quota proportional to success_rate (equal expected valid yield per radius if same seed budget).
    Alternative: equal seeds — here we scale 10k * (rate / sum(rates)).
    """
    rates = [r["success_rate"] for r in results]
    s = sum(rates)
    if s <= 0:
        return {"error": "all success rates zero"}
    raw = [target_total * (r / s) for r in rates]
    counts = [int(round(x)) for x in raw]
    counts[0] += target_total - sum(counts)
    suggested = {}
    for label, r, n in zip(STRATUM_LABELS, results, counts):
        suggested[label] = {
            "radius_km": r["radius_km"],
            "suggested_n_windows": n,
            "fraction_of_10k": round(n / target_total, 3),
            "success_rate_probe": r["success_rate"],
        }
    return suggested


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe random-seed validity by radius")
    parser.add_argument(
        "--network-dir",
        default=None,
        help="Folder with network_XX.pkl (default: auto ./network_structures)",
    )
    parser.add_argument("--seeds-per-radius", type=int, default=100)
    parser.add_argument("--n-min", type=int, default=15)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-total", type=int, default=10000, help="For suggested quota math")
    parser.add_argument("--out", default="results_sampling/probe_random_seed.json")
    args = parser.parse_args()

    network_dir = resolve_network_dir(args.network_dir)
    print(f"Loading nodes from {network_dir} ...", flush=True)
    nodes = load_national_nodes(network_dir)
    print(f"  {len(nodes):,} national nodes", flush=True)

    tree = BallTree(np.radians(nodes[["lat", "lon"]].values), metric="haversine")
    rng = np.random.default_rng(args.seed)

    results = []
    print(f"\nProbe: {args.seeds_per_radius} random seeds per radius | n_min={args.n_min}\n", flush=True)
    print(f"{'R (km)':>8} {'valid':>8} {'rate':>8} {'mean_n':>10} {'med_n(valid)':>14}")
    print("-" * 52)

    for radius_km, label in zip(RADII_KM, STRATUM_LABELS):
        r = probe_radius(
            nodes, tree, radius_km, args.seeds_per_radius, rng, args.n_min, args.n_max
        )
        r["window_stratum"] = label
        results.append(r)
        med_v = r["n_nodes_valid_median"]
        med_s = f"{med_v:.0f}" if med_v is not None else "—"
        print(
            f"{radius_km:8.0f} {r['n_valid']:8d} {r['success_rate']:7.1%} "
            f"{r['n_nodes_all_mean']:10.1f} {med_s:>14}"
        )

    suggested_equal_yield = suggest_quotas(results, args.target_total)

    # Fixed 5:3:2 reference
    ref_532 = {"urban": 5000, "suburban": 3000, "rural": 2000}

    report = {
        "probe": {
            "seeds_per_radius": args.seeds_per_radius,
            "n_min": args.n_min,
            "n_max": args.n_max,
            "seed": args.seed,
            "national_nodes": len(nodes),
        },
        "by_radius": results,
        "suggested_quota_by_success_rate": suggested_equal_yield,
        "reference_quota_5_3_2": ref_532,
        "how_to_use": (
            "If 80km success_rate is low, reduce its quota vs 10km. "
            "suggested_quota_by_success_rate splits 10k so each radius has similar "
            "expected valid windows per seed draw (rate-weighted)."
        ),
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Suggested 10k split (weighted by probe success rate) ===")
    for label, info in suggested_equal_yield.items():
        print(
            f"  {label:9s} R={info['radius_km']:.0f} km  "
            f"n={info['suggested_n_windows']:5d}  ({info['fraction_of_10k']:.1%})  "
            f"probe_rate={info['success_rate_probe']:.1%}"
        )
    print(f"\nReference 5:3:2 → {ref_532}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
