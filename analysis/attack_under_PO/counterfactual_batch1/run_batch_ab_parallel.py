#!/usr/bin/env python3
"""
Batch A / B — year-specific outage persistence tests (#1).

A: network year t × outage year t   (true co-temporal)
B: network fixed 2023 × outage year t  (weather-only fluctuation)

Reuses the same CRN engine as Batch 1. Separate output roots — never overwrites
results_mc_10km_panel_2018_2026/ or results_cf_batch1_*.

Usage:
  python analysis/attack_under_PO/counterfactual_batch1/run_batch_ab_parallel.py \\
      --design A --years 2018 2019 2020 2021 2022 2023 --n-sims 100 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATTACK = Path(__file__).resolve().parents[1]
for p in (ROOT, ATTACK):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402
from counterfactual_batch1.scenarios import Scenario  # noqa: E402

DEFAULT_NETWORK_ROOT = ROOT / "outputs" / "network_graph_10km_2018_2026"
DEFAULT_OUT_A = ROOT / "results_cf_batchA_year_matched"
DEFAULT_OUT_B = ROOT / "results_cf_batchB_net2023_outage_year"


def _default_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, min(12, n - 2))


def members_abbr_to_full_names(members: str) -> list[str]:
    names: list[str] = []
    for abbr in members.split(","):
        abbr = abbr.strip()
        if abbr:
            names.append(ABBR_TO_STATE_NAME.get(abbr, abbr))
    return names


def build_jobs(network_root: Path, years: list[int], only: set[str] | None) -> list[dict]:
    import pandas as pd

    jobs: list[dict] = []
    for year in years:
        year_dir = network_root / str(year)
        summary = year_dir / "network_summary.csv"
        net_dir = year_dir / "network_structures"
        if not summary.is_file():
            print(f"[skip] missing {summary}")
            continue
        df = pd.read_csv(summary)
        for _, row in df.iterrows():
            unit = str(row["unit"])
            display = str(row.get("display_name", unit))
            if only and unit not in only and display not in only:
                continue
            pkl = net_dir / f"network_{unit}.pkl"
            if not pkl.is_file():
                alt = net_dir / f"network_region_{unit}.pkl"
                pkl = alt if alt.is_file() else pkl
            if not pkl.is_file():
                continue
            members = str(row.get("members", unit))
            jobs.append(
                {
                    "network_year": year,
                    "outage_year": year,
                    "label": display,
                    "unit": unit,
                    "member_states": members_abbr_to_full_names(members),
                    "pkl_path": str(pkl.resolve()),
                }
            )
    return jobs


def build_jobs_B(
    network_root: Path, years: list[int], only: set[str] | None
) -> list[dict]:
    """Fixed 2023 network; vary outage year."""
    import pandas as pd

    year_dir = network_root / "2023"
    summary = year_dir / "network_summary.csv"
    net_dir = year_dir / "network_structures"
    if not summary.is_file():
        raise SystemExit(f"Missing {summary}")
    df = pd.read_csv(summary)
    jobs: list[dict] = []
    for outage_year in years:
        for _, row in df.iterrows():
            unit = str(row["unit"])
            display = str(row.get("display_name", unit))
            if only and unit not in only and display not in only:
                continue
            pkl = net_dir / f"network_{unit}.pkl"
            if not pkl.is_file():
                alt = net_dir / f"network_region_{unit}.pkl"
                pkl = alt if alt.is_file() else pkl
            if not pkl.is_file():
                continue
            members = str(row.get("members", unit))
            jobs.append(
                {
                    "network_year": 2023,
                    "outage_year": outage_year,
                    "label": display,
                    "unit": unit,
                    "member_states": members_abbr_to_full_names(members),
                    "pkl_path": str(pkl.resolve()),
                }
            )
    return jobs


def _worker(args: tuple) -> str:
    (
        design,
        network_year,
        outage_year,
        label,
        unit,
        member_states,
        pkl_path,
        out_dir,
        n_sims,
        force,
        write_events,
    ) = args

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    attack = Path(__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[3]
    for p in (str(root), str(attack)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from counterfactual_batch1.crn_engine import run_unit_batch1
    from counterfactual_batch1.scenarios import Scenario

    # Only baseline for A/B persistence tests
    scenarios = [
        Scenario(
            key="baseline",
            family="baseline",
            description=f"design={design} net={network_year} outage={outage_year}",
        )
    ]
    # Nest under out/<outage_year>/<unit> so years don't collide
    nested = Path(out_dir) / str(outage_year)
    densify_cache = root / "outputs" / "network_graph_cf_densify_2023"  # unused for baseline

    return run_unit_batch1(
        label=label,
        unit=unit,
        member_states=member_states,
        pkl_path=Path(pkl_path),
        out_dir=nested,
        scenarios=scenarios,
        n_sims=n_sims,
        project_root=root,
        densify_cache=densify_cache,
        k5km_pkl=None,
        bootstrap_pool="all",
        outage_year=int(outage_year),
        force=force,
        write_events=write_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CF Batch A/B year-specific outages")
    parser.add_argument("--design", choices=("A", "B"), required=True)
    parser.add_argument("--network-root", type=str, default=str(DEFAULT_NETWORK_ROOT))
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2018, 2024)))
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    network_root = Path(args.network_root)
    if not network_root.is_absolute():
        network_root = ROOT / network_root
    out_root = Path(args.out) if args.out else (
        DEFAULT_OUT_A if args.design == "A" else DEFAULT_OUT_B
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    if args.design == "A":
        jobs = build_jobs(network_root, args.years, only)
    else:
        jobs = build_jobs_B(network_root, args.years, only)

    workers = args.workers if args.workers is not None else _default_workers()
    (out_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "design": args.design,
                "years": args.years,
                "n_sims": args.n_sims,
                "n_jobs": len(jobs),
                "note": "Year-specific outage severity pool; λ still from pooled ADVI",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[CF-{args.design}] jobs={len(jobs)} workers={workers} out={out_root}")

    task_args = [
        (
            args.design,
            j["network_year"],
            j["outage_year"],
            j["label"],
            j["unit"],
            j["member_states"],
            j["pkl_path"],
            str(out_root),
            args.n_sims,
            args.force,
            not args.no_events,
        )
        for j in jobs
    ]

    t0 = time.time()
    if workers <= 1:
        for a in task_args:
            print(_worker(a), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, a) for a in task_args]
            for fut in as_completed(futs):
                print(fut.result(), flush=True)
    print(f"[CF-{args.design}] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
