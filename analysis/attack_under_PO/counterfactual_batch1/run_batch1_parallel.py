#!/usr/bin/env python3
"""Run the 2023 corridor-expansion and severity-reduction experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis"
ATTACK = Path(__file__).resolve().parents[1]
for path in (ROOT, ANALYSIS, ATTACK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from counterfactual_batch1.scenarios import batch1_scenarios  # noqa: E402
from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402

NETWORK_ROOT = ROOT / "outputs" / "1029-network-data-preprocessing-10km-geodesic-2018-2023"
CACHE_ROOT = ROOT / "outputs" / "1029-network-nevi-expansion-2023"
OUTPUT_ROOT = ROOT / "results_interventions_1029_network2023_pooled2018_2023"


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _member_names(members: str) -> list[str]:
    return [ABBR_TO_STATE_NAME.get(x.strip(), x.strip()) for x in members.split(",") if x.strip()]


def _network_jobs(network_root: Path, year: int, selected: set[str]) -> list[dict]:
    year_dir = network_root / str(year)
    summary = pd.read_csv(year_dir / "network_summary.csv")
    jobs = []
    for row in summary.to_dict("records"):
        unit = str(row["unit"])
        label = str(row.get("display_name", unit))
        if selected and unit not in selected and label not in selected:
            continue
        graph_path = year_dir / "network_structures" / f"network_{unit}.pkl"
        if not graph_path.is_file():
            graph_path = year_dir / "network_structures" / f"network_region_{unit}.pkl"
        if not graph_path.is_file():
            raise FileNotFoundError(f"Missing network for {unit}")
        jobs.append(
            {
                "unit": unit,
                "label": label,
                "member_states": _member_names(str(row.get("members", unit))),
                "pkl_path": graph_path.resolve(),
                "n_nodes": int(row["n_nodes"]) if pd.notna(row.get("n_nodes")) else 0,
            }
        )
    return sorted(jobs, key=lambda job: (-job["n_nodes"], job["unit"]))


def _preflight(jobs: list[dict], deep: bool) -> None:
    from counterfactual_batch1.densify_network import resolve_interstate_shapefile
    from monte_carlo_outage_attack import load_network

    required = {
        "posterior": ROOT / "advi_results_2018_2023" / "county_advi_posterior.nc",
        "county rates": ROOT / "advi_results_2018_2023" / "county_estimates_advi.csv",
        "outage events": ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv",
        "customer counts": ROOT / "notebooks" / "PO_data_cleaning" / "MCC.csv",
        "coverage history": ROOT.parent / "24237376" / "coverage_history.csv",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    population = (
        ROOT / "data" / "processed" / "pop_units_epicenter.gpkg",
        ROOT / "data" / "processed" / "pop_units_epicenter.parquet",
    )
    if not any(path.is_file() for path in population):
        missing.append("population epicenters: " + " or ".join(map(str, population)))
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))

    resolve_interstate_shapefile()
    errors = []
    for job in jobs:
        graph = load_network(job["pkl_path"])
        if graph.number_of_nodes() != job["n_nodes"]:
            errors.append(f"{job['unit']}: node count differs from network_summary.csv")
        if graph.number_of_nodes() < 2 or not nx.is_connected(graph.to_undirected()):
            errors.append(f"{job['unit']}: network is empty or disconnected")
    if errors:
        raise ValueError("Invalid networks:\n  " + "\n  ".join(errors))

    if not deep:
        return
    from compute_impact_radius import load_county_geometry_and_mcc, load_coverage_history
    from sample_lambda_from_posterior import load_posterior
    from select_epicenter_by_population import require_pop_units

    counties, posterior = load_posterior()
    if posterior.ndim != 2 or posterior.shape[1] != len(counties):
        raise ValueError("Posterior county dimension does not match the county-rate table")
    tables = (
        pd.read_csv(required["outage events"]),
        load_county_geometry_and_mcc(ROOT),
        load_coverage_history(ROOT),
        require_pop_units(ROOT),
    )
    if any(table.empty for table in tables):
        raise ValueError("At least one required input table is empty")
    print(
        f"PASS: {len(jobs)} networks, {posterior.shape[0]} posterior draws, "
        f"{posterior.shape[1]} counties, and {len(tables[0])} events"
    )


def _run_job(task: dict) -> str:
    from counterfactual_batch1.crn_engine import run_unit_batch1

    return run_unit_batch1(
        label=task["label"],
        unit=task["unit"],
        member_states=task["member_states"],
        pkl_path=task["pkl_path"],
        out_dir=task["out"],
        scenarios=batch1_scenarios(),
        n_sims=task["n_sims"],
        project_root=ROOT,
        densify_cache=task["cache"],
        force=task["force"],
        write_events=task["write_events"],
    )


def _write_panel(out: Path) -> None:
    summaries = [pd.read_csv(path) for path in sorted(out.glob("*/summary.csv"))]
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            out / "panel_scenario_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-root", type=Path, default=NETWORK_ROOT)
    parser.add_argument("--network-year", type=int, default=2023)
    parser.add_argument("--densify-cache", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", default="", help="Comma-separated analytical units")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.n_sims < 1 or args.workers < 1:
        raise ValueError("--n-sims and --workers must be positive")
    network_root, cache, out = map(
        _absolute, (args.network_root, args.densify_cache, args.out)
    )
    selected = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = _network_jobs(network_root, args.network_year, selected)
    expected = len(selected) if selected else 45
    if not jobs or (not selected and len(jobs) != expected):
        raise ValueError(f"Expected {expected} analytical units, found {len(jobs)}")
    _preflight(jobs, deep=args.check_only)
    if args.check_only:
        return

    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(jobs))
    scenarios = batch1_scenarios()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "network_root": str(network_root),
        "network_year": args.network_year,
        "outage_model": "pooled 2018-2023 posterior and event pool",
        "state_rate": "sum of county rates",
        "county_assignment": "county rate / state rate",
        "epicenter": "census-tract population weighted",
        "event_count": "Poisson draw from the summed county rates",
        "event_reset": "restore the original scenario network after every event",
        "global_efficiency": "unweighted, with the original-node denominator fixed after removal",
        "corridor_intervention": "50-mile Interstate candidates, at least 10 km apart",
        "severity_intervention": "reduce affected customers, then recompute capped radius",
        "scenarios": [scenario.key for scenario in scenarios],
        "n_sims": args.n_sims,
        "workers": workers,
        "write_event_files": not args.no_events,
    }
    (out / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tasks = [
        {
            **job,
            "out": out,
            "cache": cache,
            "n_sims": args.n_sims,
            "force": args.force,
            "write_events": not args.no_events,
        }
        for job in jobs
    ]
    print(
        f"[Interventions] units={len(tasks)} scenarios={len(scenarios)} "
        f"sims={args.n_sims} workers={workers}\nout={out}",
        flush=True,
    )

    started = time.time()
    results = []
    if workers == 1:
        for task in tasks:
            result = _run_job(task)
            results.append(result)
            print(result, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_job, task): task["unit"] for task in tasks}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = f"error:{unit}:{error}"
                results.append(result)
                print(result, flush=True)

    _write_panel(out)
    failures = [result for result in results if not result.startswith(("ok:", "skip:"))]
    print(
        f"[Interventions] done ok={sum(r.startswith('ok:') for r in results)} "
        f"skip={sum(r.startswith('skip:') for r in results)} "
        f"failed={len(failures)} in {(time.time() - started) / 60:.1f} min"
    )
    if failures:
        raise RuntimeError("Incomplete units:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
