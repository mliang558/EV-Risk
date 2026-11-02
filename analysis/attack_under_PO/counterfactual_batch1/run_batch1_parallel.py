#!/usr/bin/env python3
"""Run the 2023 corridor, demand-targeted, and severity interventions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import networkx as nx
import numpy as np
import pandas as pd

def _project_root() -> Path:
    marker = Path("analysis/attack_under_PO")
    for candidate in (Path.cwd(), Path(__file__).absolute().parents[3]):
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(f"Run from the project checkout containing {marker}")


ROOT = _project_root()
ANALYSIS = ROOT / "analysis"
ATTACK = ROOT / "analysis" / "attack_under_PO"
for path in (ROOT, ANALYSIS, ATTACK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from counterfactual_batch1.scenarios import batch1_scenarios  # noqa: E402
from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402

NETWORK_ROOT = ROOT / "outputs" / "1029-network-data-preprocessing-10km-geodesic-2018-2023"
CACHE_ROOT = ROOT / "outputs" / "1031-network-expansion-comparators-2023"
OUTPUT_ROOT = ROOT / "results_interventions_1031_network2023_pooled2018_2023"
POPULATION_UNITS = ROOT / "data" / "processed" / "tract_population_units_2020.gpkg"


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                "pkl_path": graph_path.absolute(),
                "n_nodes": int(row["n_nodes"]) if pd.notna(row.get("n_nodes")) else 0,
            }
        )
    return sorted(jobs, key=lambda job: (-job["n_nodes"], job["unit"]))


def _preflight(
    jobs: list[dict],
    deep: bool,
    families: set[str],
    epicenter_mode: str,
    population_path: Path,
) -> dict:
    from counterfactual_batch1.densify_network import (
        validate_ev_afc_file,
        validate_primary_roads_file,
    )
    from monte_carlo_outage_attack import load_network

    required = {
        "posterior": ROOT / "advi_results_2018_2023" / "county_advi_posterior.nc",
        "county rates": ROOT / "advi_results_2018_2023" / "county_estimates_advi.csv",
        "outage events": ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv",
        "customer counts": ROOT / "notebooks" / "PO_data_cleaning" / "MCC.csv",
        "coverage history": ROOT.parent / "24237376" / "coverage_history.csv",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    need_population = epicenter_mode == "population" or "Demand" in families
    if need_population and not population_path.is_file():
        missing.append(f"census-tract population: {population_path}")
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))

    sources = {}
    if population_path.is_file():
        sources["population"] = {
            "path": str(population_path.resolve()),
            "sha256": _sha256(population_path),
            "vintage": "2020 Decennial Census P1",
        }
    if families & {"CF-D", "Demand"}:
        sources["afc"] = validate_ev_afc_file()
        afc = sources["afc"]
        print(
            f"OK: official EV AFC corridors: {afc['n_segments']} segments, "
            f"statuses={afc['statuses']}\n    {afc['path']}"
        )
    if "Demand" in families:
        sources["primary_roads"] = validate_primary_roads_file()
        roads = sources["primary_roads"]
        print(f"OK: Census primary roads ({roads['vintage']}): {roads['path']}")
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
        return sources
    from compute_impact_radius import (
        STATE_NAME_TO_ABBR,
        load_county_geometry_and_mcc,
        load_coverage_history,
    )
    from monte_carlo_outage_attack import load_county_polygons
    from sample_lambda_from_posterior import load_posterior, prepare_outages_for_bootstrap
    counties, posterior = load_posterior()
    if posterior.ndim != 2 or posterior.shape[1] != len(counties):
        raise ValueError("Posterior county dimension does not match the county-rate table")
    outages = prepare_outages_for_bootstrap(pd.read_csv(required["outage events"]))
    county_info = load_county_geometry_and_mcc(ROOT)
    coverage = load_coverage_history(ROOT)
    if need_population:
        import geopandas as gpd

        population = gpd.read_file(population_path)
    else:
        population = None
    polygons = load_county_polygons(ROOT)
    tables = (outages, county_info, coverage, polygons)
    if population is not None:
        tables = (*tables, population)
    if any(table.empty for table in tables):
        raise ValueError("At least one required input table is empty")

    target_states = {state for job in jobs for state in job["member_states"]}
    missing_states = target_states - set(counties["state"])
    missing_pools = target_states - set(outages["state"])
    if missing_states or missing_pools:
        raise ValueError(
            f"Missing posterior states={sorted(missing_states)}; "
            f"missing historical pools={sorted(missing_pools)}"
        )

    target_counties = counties[counties["state"].isin(target_states)].copy()
    target_fips = set(
        target_counties["fips"].astype(int).astype(str).str.zfill(5)
    )
    county_info = county_info.set_index("fips_str").reindex(sorted(target_fips))
    invalid_counties = county_info[
        county_info[["MCC", "rho", "area_km2"]].isna().any(axis=1)
        | (county_info["MCC"] <= 0)
        | (county_info["rho"] <= 0)
        | (county_info["area_km2"] <= 0)
    ].index.tolist()
    polygon_fips = set(polygons["fips_str"].astype(str).str.zfill(5))
    missing_polygons = sorted(target_fips - polygon_fips)
    if invalid_counties or missing_polygons:
        raise ValueError(
            f"Invalid MCC/area counties={invalid_counties[:10]}; "
            f"missing county polygons={missing_polygons[:10]}"
        )

    relevant = outages[outages["state"].isin(target_states)].copy()
    customer_col = "mean_customers" if "mean_customers" in relevant else "affected_customers"
    invalid_events = (
        ~np.isfinite(relevant["duration_hours"].astype(float))
        | (relevant["duration_hours"].astype(float) < 0)
        | ~np.isfinite(relevant[customer_col].astype(float))
        | (relevant[customer_col].astype(float) < 0)
    )
    if invalid_events.any():
        count = int(invalid_events.sum())
        raise ValueError(f"Historical event pool contains {count} invalid rows")

    if "year" not in relevant.columns:
        relevant["year"] = pd.to_datetime(relevant["start_time"]).dt.year
    relevant["state_abbr"] = relevant["state"].map(STATE_NAME_TO_ABBR)
    coverage_keys = set(zip(coverage["state_abbr"], coverage["year_num"].astype(int)))
    event_keys = set(zip(relevant["state_abbr"], relevant["year"].astype(int)))
    missing_coverage = sorted(event_keys - coverage_keys, key=str)
    if missing_coverage:
        raise ValueError(f"Missing state-year coverage rows: {missing_coverage[:10]}")

    missing_population: list[str] = []
    if population is not None:
        pop_fips = set(
            population.loc[population["population"] > 0, "county_fips"]
            .astype(str)
            .str.zfill(5)
        )
        missing_population = sorted(target_fips - pop_fips)
    if need_population and missing_population:
        raise ValueError(
            "Positive-population tracts are required for every modeled "
            f"county; missing={missing_population[:10]}"
        )
    print(
        f"PASS: {len(jobs)} networks, {posterior.shape[0]} posterior draws, "
        f"{posterior.shape[1]} counties, {len(outages)} events, "
        f"epicenter={epicenter_mode}, and complete required population coverage"
    )
    return sources


def _run_job(task: dict) -> str:
    from counterfactual_batch1.crn_engine import run_unit_batch1

    return run_unit_batch1(
        label=task["label"],
        unit=task["unit"],
        member_states=task["member_states"],
        pkl_path=task["pkl_path"],
        out_dir=task["out"],
        scenarios=task["scenarios"],
        n_sims=task["n_sims"],
        project_root=ROOT,
        densify_cache=task["cache"],
        force=task["force"],
        write_events=task["write_events"],
        epicenter_mode=task["epicenter_mode"],
        population_units_path=task["population_units"],
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
    parser.add_argument("--population-units", type=Path, default=POPULATION_UNITS)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", default="", help="Comma-separated analytical units")
    parser.add_argument(
        "--families",
        default="baseline,CF-D,Demand,CF-S",
        help="Comma-separated scenario families",
    )
    parser.add_argument(
        "--epicenter-mode",
        choices=("county_uniform", "population", "station"),
        default="county_uniform",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.n_sims < 1 or args.workers < 1:
        raise ValueError("--n-sims and --workers must be positive")
    network_root, cache, out, population_units = map(
        _absolute,
        (args.network_root, args.densify_cache, args.out, args.population_units),
    )
    selected = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = _network_jobs(network_root, args.network_year, selected)
    expected = len(selected) if selected else 45
    if not jobs or (not selected and len(jobs) != expected):
        raise ValueError(f"Expected {expected} analytical units, found {len(jobs)}")
    requested_families = {
        family.strip() for family in args.families.split(",") if family.strip()
    }
    scenarios = [
        scenario
        for scenario in batch1_scenarios()
        if scenario.family in requested_families
    ]
    if not scenarios:
        raise ValueError(f"No scenarios matched --families={args.families!r}")
    sources = _preflight(
        jobs,
        deep=args.check_only,
        families=requested_families,
        epicenter_mode=args.epicenter_mode,
        population_path=population_units,
    )
    if args.check_only:
        return

    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(jobs))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "network_root": str(network_root),
        "network_year": args.network_year,
        "outage_model": "pooled 2018-2023 posterior and event pool",
        "state_rate": "sum of county rates",
        "state_rate_definition": "Lambda_s = sum_c lambda_c",
        "county_assignment": "county rate / state rate",
        "epicenter": args.epicenter_mode,
        "epicenter_definition": (
            "uniform random point within the selected county polygon; shared across "
            "baseline and intervention scenarios"
            if args.epicenter_mode == "county_uniform"
            else args.epicenter_mode
        ),
        "event_count": "Poisson draw from the summed county rates",
        "coverage_adjustment": (
            "state-year max_pct_covered; 2023 carries forward each state's 2022 value"
        ),
        "event_reset": "restore the original scenario network after every event",
        "global_efficiency": "unweighted, with the original-node denominator fixed after removal",
        "primary_intervention_metric": (
            "loss among the original 2023 hypernodes; added nodes may serve as paths "
            "but are excluded from the endpoint set and denominator"
        ),
        "corridor_intervention": (
            "FHWA-designated EV Alternative Fuel Corridor candidates sampled every "
            "50 miles and kept at least 10 km from all existing and added hypernodes"
        ),
        "corridor_source": sources.get("afc"),
        "demand_comparator": (
            "add exactly the corridor scenario's realized number of nodes, prioritizing "
            "high-population census tracts; each candidate lies on a Census primary "
            "road inside its tract and follows the same 10 km separation rule"
        ),
        "demand_road_source": sources.get("primary_roads"),
        "demand_population_source": sources.get("population"),
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
            "scenarios": scenarios,
            "epicenter_mode": args.epicenter_mode,
            "population_units": population_units,
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
