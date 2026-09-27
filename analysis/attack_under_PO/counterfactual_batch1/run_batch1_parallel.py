#!/usr/bin/env python3
"""
Parallel launcher for Counterfactual Batch 1 (2023 × pooled outages).

Does NOT write into results_mc_10km_panel_2018_2026/.
Default out: results_cf_batch1_2023_pooled/

Usage (cluster):
  export PYTHONPATH="$ROOT/analysis:$ROOT/analysis/attack_under_PO"
  python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \\
      --n-sims 100 --workers 12 --families baseline,CF-D,CF-S

Priority families: baseline,CF-D (NEVI corridor),CF-S
# Optional: CF-D-pop, CF-D-null (10 reps), CF-T, U, K
# Base nets: reuse outputs/network_graph_10km_2018_2026/2023 (do not rebuild).
# Epicenters locked to baseline G; CF nodes never chosen as epicenter.
Optional later: CF-T,U,K
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
PKG = Path(__file__).resolve().parent
for p in (ROOT, ATTACK, PKG.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402
from counterfactual_batch1.scenarios import batch1_scenarios, filter_scenarios  # noqa: E402

DEFAULT_NETWORK_ROOT = ROOT / "outputs" / "network_graph_10km_2018_2026"
DEFAULT_K5KM_ROOT = ROOT / "outputs" / "network_graph_5km_2018_2026"
DEFAULT_OUT = ROOT / "results_cf_batch1_2023_pooled"
DEFAULT_DENSIFY_CACHE = ROOT / "outputs" / "network_graph_cf_densify_2023"
NETWORK_YEAR = 2023


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


def build_jobs(
    network_root: Path,
    only_units: set[str] | None,
) -> list[dict]:
    year_dir = network_root / str(NETWORK_YEAR)
    summary_path = year_dir / "network_summary.csv"
    net_dir = year_dir / "network_structures"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")
    summary = pd_read(summary_path)
    jobs: list[dict] = []
    for _, row in summary.iterrows():
        unit = str(row["unit"])
        display = str(row.get("display_name", unit))
        if only_units and unit not in only_units and display not in only_units:
            continue
        pkl = net_dir / f"network_{unit}.pkl"
        if not pkl.is_file():
            # merged regions sometimes use network_region_*.pkl
            alt = net_dir / f"network_region_{unit}.pkl"
            pkl = alt if alt.is_file() else pkl
        if not pkl.is_file():
            print(f"[skip] missing pickle for {unit}")
            continue
        members = str(row.get("members", unit))
        jobs.append(
            {
                "label": display,
                "unit": unit,
                "member_states": members_abbr_to_full_names(members),
                "pkl_path": str(pkl.resolve()),
            }
        )
    return jobs


def pd_read(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _worker(args: tuple) -> str:
    (
        label,
        unit,
        member_states,
        pkl_path,
        out_dir,
        n_sims,
        densify_cache,
        k5km_root,
        families,
        keys,
        force,
        write_events,
        bootstrap_pool,
    ) = args

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    attack = Path(__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[3]
    for p in (str(root), str(attack), str(attack)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from counterfactual_batch1.crn_engine import run_unit_batch1
    from counterfactual_batch1.scenarios import batch1_scenarios, filter_scenarios

    fam = {x.strip() for x in families.split(",") if x.strip()} or None
    keyset = {x.strip() for x in keys.split(",") if x.strip()} or None
    scenarios = filter_scenarios(batch1_scenarios(), fam, keyset)

    k5 = Path(k5km_root) / str(NETWORK_YEAR) / "network_structures" / f"network_{unit}.pkl"
    if not k5.is_file():
        k5 = None

    return run_unit_batch1(
        label=label,
        unit=unit,
        member_states=member_states,
        pkl_path=Path(pkl_path),
        out_dir=Path(out_dir),
        scenarios=scenarios,
        n_sims=n_sims,
        project_root=root,
        densify_cache=Path(densify_cache),
        k5km_pkl=k5,
        bootstrap_pool=bootstrap_pool,
        outage_year=None,  # pooled
        force=force,
        write_events=write_events,
    )


def _write_panel(out_root: Path) -> None:
    import pandas as pd

    rows = []
    for p in sorted(out_root.glob("*/summary.csv")):
        rows.append(pd.read_csv(p))
    if rows:
        panel = pd.concat(rows, ignore_index=True)
        panel.to_csv(out_root / "panel_scenario_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="CF Batch1 parallel runner (2023×pooled)")
    parser.add_argument("--network-root", type=str, default=str(DEFAULT_NETWORK_ROOT))
    parser.add_argument("--k5km-root", type=str, default=str(DEFAULT_K5KM_ROOT))
    parser.add_argument("--densify-cache", type=str, default=str(DEFAULT_DENSIFY_CACHE))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", type=str, default="", help="Comma-separated units")
    parser.add_argument(
        "--families",
        type=str,
        default="baseline,CF-D,CF-S",
        help="Comma families: baseline,CF-D,CF-D-pop,CF-D-null,CF-S,CF-T,U,K",
    )
    parser.add_argument("--keys", type=str, default="", help="Optional explicit scenario keys")
    parser.add_argument("--bootstrap-pool", choices=("all", "severe_top10"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-events", action="store_true", help="Skip per-event gz dumps")
    args = parser.parse_args()

    network_root = Path(args.network_root)
    if not network_root.is_absolute():
        network_root = ROOT / network_root
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    densify_cache = Path(args.densify_cache)
    if not densify_cache.is_absolute():
        densify_cache = ROOT / densify_cache
    k5km_root = Path(args.k5km_root)
    if not k5km_root.is_absolute():
        k5km_root = ROOT / k5km_root

    out_root.mkdir(parents=True, exist_ok=True)
    densify_cache.mkdir(parents=True, exist_ok=True)

    only_set = {x.strip() for x in args.only.split(",") if x.strip()} or None
    jobs = build_jobs(network_root, only_set)
    if not jobs:
        raise SystemExit("No jobs.")

    workers = args.workers if args.workers is not None else _default_workers()
    fam = {x.strip() for x in args.families.split(",") if x.strip()} or None
    keyset = {x.strip() for x in args.keys.split(",") if x.strip()} or None
    sc = filter_scenarios(batch1_scenarios(), fam, keyset)

    manifest = {
        "batch": "cf_batch1_2023_pooled",
        "network_year": NETWORK_YEAR,
        "outage_pool": "pooled_2018_2023",
        "n_sims": args.n_sims,
        "n_units": len(jobs),
        "scenarios": [s.key for s in sc],
        "network_root": str(network_root),
        "out": str(out_root),
    }
    (out_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[CF-B1] units={len(jobs)} scenarios={len(sc)} "
        f"n_sims={args.n_sims} workers={workers}"
    )
    print(f"  out -> {out_root}")
    print(f"  scenarios: {', '.join(s.key for s in sc)}")

    task_args = [
        (
            j["label"],
            j["unit"],
            j["member_states"],
            j["pkl_path"],
            str(out_root),
            args.n_sims,
            str(densify_cache),
            str(k5km_root),
            args.families,
            args.keys,
            args.force,
            not args.no_events,
            args.bootstrap_pool,
        )
        for j in jobs
    ]

    results: list[str] = []
    t0 = time.time()
    if workers <= 1:
        for a in task_args:
            results.append(_worker(a))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, a): a[1] for a in task_args}
            for fut in as_completed(futs):
                unit = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = f"error:{unit}:{e}"
                results.append(r)
                print(r, flush=True)

    _write_panel(out_root)
    ok = sum(1 for r in results if r.startswith("ok:"))
    skip = sum(1 for r in results if r.startswith("skip:"))
    print(
        f"[CF-B1] done ok={ok} skip={skip} other={len(results)-ok-skip} "
        f"in {(time.time()-t0)/60:.1f} min"
    )


if __name__ == "__main__":
    main()
