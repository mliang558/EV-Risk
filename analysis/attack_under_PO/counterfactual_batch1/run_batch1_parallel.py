#!/usr/bin/env python3
"""
Parallel launcher for Counterfactual Batch 1 (2023 × pooled outages).

Does NOT write into results_mc_10km_panel_2018_2026/.
Default out: results_interventions_1029_network2023_pooled2018_2023/

Usage (cluster):
  export PYTHONPATH="$ROOT/analysis:$ROOT/analysis/attack_under_PO"
  python analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \\
      --n-sims 100 --workers 12 --families baseline,CF-D,CF-S

Priority families: baseline,CF-D (NEVI),CF-S (N_affected),U,K
# Requires: data/processed/pop_units_epicenter.gpkg (census-tract epicenters)
# Optional: CF-D-pop, CF-D-null; EPICENTER_MODE=station only for appendix ablation
# Base nets: reuse the validated 1029 canonical network for 2023.
# Epicenters: census-tract population only (never station-KDE).
Optional appendix controls: CF-D-pop,CF-D-null
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One analytical unit is assigned to each process. Keep numerical libraries
# single-threaded so --workers controls the total CPU allocation.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[3]
ATTACK = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
for p in (ROOT, ATTACK, PKG.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402
from counterfactual_batch1.scenarios import batch1_scenarios, filter_scenarios  # noqa: E402

DEFAULT_NETWORK_ROOT = (
    ROOT / "outputs" / "1029-network-data-preprocessing-10km-geodesic-2018-2023"
)
DEFAULT_K5KM_ROOT = ROOT / "outputs" / "network_graph_5km_2018_2026"
DEFAULT_OUT = ROOT / "results_interventions_1029_network2023_pooled2018_2023"
DEFAULT_DENSIFY_CACHE = ROOT / "outputs" / "1029-network-nevi-expansion-2023"
NETWORK_YEAR = 2023
RUN_LABEL = "Intervention experiments"


def _default_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, min(12, n - 2))


def _git_rev() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def members_abbr_to_full_names(members: str) -> list[str]:
    names: list[str] = []
    for abbr in members.split(","):
        abbr = abbr.strip()
        if abbr:
            names.append(ABBR_TO_STATE_NAME.get(abbr, abbr))
    return names


def build_jobs(
    network_root: Path,
    network_year: int,
    only_units: set[str] | None,
) -> list[dict]:
    year_dir = network_root / str(network_year)
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
        n_nodes = int(row["n_nodes"]) if "n_nodes" in row and pd_notna(row["n_nodes"]) else 0
        jobs.append(
            {
                "label": display,
                "unit": unit,
                "member_states": members_abbr_to_full_names(members),
                "pkl_path": str(pkl.resolve()),
                "n_nodes": n_nodes,
            }
        )
    # Largest networks first so CA/NY/TX start immediately (long-tail makespan)
    jobs.sort(key=lambda j: (-int(j.get("n_nodes", 0)), j["unit"]))
    return jobs


def validate_network_jobs(jobs: list[dict]) -> None:
    """Load every selected canonical graph and verify its summary metadata."""
    import networkx as nx

    from monte_carlo_outage_attack import load_network

    errors: list[str] = []
    for job in jobs:
        try:
            graph = load_network(Path(job["pkl_path"]))
        except Exception as exc:
            errors.append(f"{job['unit']}: cannot load {job['pkl_path']}: {exc}")
            continue
        expected_nodes = int(job.get("n_nodes", 0))
        if expected_nodes and graph.number_of_nodes() != expected_nodes:
            errors.append(
                f"{job['unit']}: graph has {graph.number_of_nodes()} nodes; "
                f"summary reports {expected_nodes}"
            )
        if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
            errors.append(f"{job['unit']}: graph is too small for efficiency analysis")
        elif not nx.is_connected(graph.to_undirected()):
            errors.append(f"{job['unit']}: baseline graph is disconnected")
    if errors:
        raise ValueError("Invalid canonical network inputs:\n  " + "\n  ".join(errors))


def pd_notna(x) -> bool:
    try:
        import pandas as pd

        return bool(pd.notna(x))
    except Exception:
        return x is not None


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
        epicenter_mode,
        network_year,
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

    k5 = Path(k5km_root) / str(network_year) / "network_structures" / f"network_{unit}.pkl"
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
        epicenter_mode=epicenter_mode,
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
    parser = argparse.ArgumentParser(
        description=(
            "Compare 2023 baseline, NEVI-style corridor expansion, and "
            "outage-severity reduction using shared pooled-outage draws."
        )
    )
    parser.add_argument("--network-root", type=str, default=str(DEFAULT_NETWORK_ROOT))
    parser.add_argument("--network-year", type=int, default=NETWORK_YEAR)
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
        help="Comma families: baseline,CF-D,CF-D-pop,CF-D-null,CF-S",
    )
    parser.add_argument("--keys", type=str, default="", help="Optional explicit scenario keys")
    parser.add_argument("--bootstrap-pool", choices=("all", "severe_top10"), default="all")
    parser.add_argument(
        "--epicenter-mode",
        type=str,
        default="population",
        choices=("population", "station"),
        help="population=census-tract (default); station=legacy appendix ablation",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-events", action="store_true", help="Skip per-event gz dumps")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate required inputs and exit without running simulations.",
    )
    args = parser.parse_args()

    if args.n_sims < 1:
        raise ValueError("--n-sims must be at least 1")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")

    fam = {x.strip() for x in args.families.split(",") if x.strip()} or None
    keyset = {x.strip() for x in args.keys.split(",") if x.strip()} or None
    sc = filter_scenarios(batch1_scenarios(), fam, keyset)

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

    required = {
        "2023 network summary": network_root / str(args.network_year) / "network_summary.csv",
        "county-rate posterior": ROOT / "advi_results_2018_2023" / "county_advi_posterior.nc",
        "county-rate table": ROOT / "advi_results_2018_2023" / "county_estimates_advi.csv",
        "cleaned outage events": (
            ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
        ),
        "county customer counts": ROOT / "notebooks" / "PO_data_cleaning" / "MCC.csv",
        "coverage history": ROOT.parent / "24237376" / "coverage_history.csv",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    population_candidates = (
        ROOT / "data" / "processed" / "pop_units_epicenter.gpkg",
        ROOT / "data" / "processed" / "pop_units_epicenter.parquet",
    )
    if not any(path.is_file() for path in population_candidates):
        missing.append(
            "population epicenters: " + " or ".join(str(path) for path in population_candidates)
        )
    if missing:
        raise FileNotFoundError("Missing intervention inputs:\n  " + "\n  ".join(missing))
    if any(s.family == "CF-D" for s in sc):
        from counterfactual_batch1.densify_network import resolve_interstate_shapefile

        interstate_path = resolve_interstate_shapefile()
        print(f"OK  Interstate corridors: {interstate_path}")

    from compute_impact_radius import resolve_county_shapefile

    county_path = resolve_county_shapefile(ROOT)
    print(f"OK  county polygons: {county_path}")

    only_set = {x.strip() for x in args.only.split(",") if x.strip()} or None
    jobs = build_jobs(network_root, args.network_year, only_set)
    if not jobs:
        raise SystemExit("No jobs.")
    if only_set is None and len(jobs) != 45:
        raise ValueError(f"Expected 45 analytical units, found {len(jobs)}")
    validate_network_jobs(jobs)
    print(f"OK  canonical network: {network_root / str(args.network_year)}")
    print(f"OK  loadable connected network graphs: {len(jobs)}")
    print("OK  pooled Bayesian rates and 2018-2023 historical event pool")
    if args.check_only:
        from compute_impact_radius import load_county_geometry_and_mcc, load_coverage_history
        from sample_lambda_from_posterior import load_posterior
        from select_epicenter_by_population import require_pop_units

        county_df, flat_lam = load_posterior()
        if flat_lam.ndim != 2 or flat_lam.shape[1] != len(county_df):
            raise ValueError(
                f"Posterior shape {flat_lam.shape} does not match "
                f"county table length {len(county_df)}"
            )
        outages = pd_read(required["cleaned outage events"])
        if "state" not in outages.columns:
            raise ValueError("Cleaned outage file lacks state")
        if not {"fips", "fips_str"}.intersection(outages.columns):
            raise ValueError("Cleaned outage file lacks county FIPS")
        if not {"mean_customers", "affected_customers"}.intersection(outages.columns):
            raise ValueError("Cleaned outage file lacks affected-customer severity")
        if not {"duration_hours", "duration_min"}.intersection(outages.columns):
            raise ValueError("Cleaned outage file lacks event duration")
        county_geometry = load_county_geometry_and_mcc(ROOT)
        coverage = load_coverage_history(ROOT)
        pop_units = require_pop_units(ROOT)
        if county_geometry.empty or coverage.empty or pop_units.empty or outages.empty:
            raise ValueError("One or more required intervention datasets are empty")
        print(
            f"OK  posterior draws={flat_lam.shape[0]} counties={flat_lam.shape[1]}; "
            f"cleaned events={len(outages)}"
        )
        print(f"OK  selected scenarios: {', '.join(s.key for s in sc)}")
        print("PASS: intervention inputs are complete")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    densify_cache.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else _default_workers()

    manifest = {
        "batch": "network_year_pooled_outages",
        "analysis_role": "prospective_counterfactual_intervention",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "network_year": args.network_year,
        "outage_pool": "pooled_2018_2023",
        "posterior_file": "advi_results_2018_2023/county_advi_posterior.nc",
        "county_metadata_file": "advi_results_2018_2023/county_estimates_advi.csv",
        "state_rate_definition": "Lambda_s = sum_c lambda_c",
        "county_assignment": "P(c|s) = lambda_c / Lambda_s",
        "event_bootstrap": "with replacement from selected county; state pool fallback",
        "epicenter_mode": args.epicenter_mode,
        "shortest_path": "unweighted hop count",
        "efficiency_denominator": "fixed baseline N*(N-1) after node removal",
        "event_loss": "((E0-E_after)/E0) * duration_hours",
        "annual_loss_W": "sum(event_loss)",
        "reported_L_tilde": "annual_loss_W / scenario_n_nodes",
        "cf_s_definition": (
            "N_affected *= (1-x), then recompute coverage-adjusted, "
            "MCC-capped disruption radius"
        ),
        "cf_d_definition": (
            "strict Interstate-corridor placement at 50-mile candidates; "
            "no off-corridor population fill; shortfalls retained"
        ),
        "n_sims": args.n_sims,
        "workers_requested": int(workers),
        "workers_used": int(max(1, min(int(workers), len(jobs)))),
        "n_units": len(jobs),
        "scenarios": [s.key for s in sc],
        "network_root": str(network_root),
        "out": str(out_root),
        "git_commit": _git_rev(),
        "no_events": bool(args.no_events),
        "metrics_include_P_hit": True,
        "note_no_events": (
            "Event gz optional; metrics_*.csv still has n_events, n_hit_events, "
            "P_hit, L_event, E_loss_given_hit, total_rel_loss_x_duration, "
            "total_rel_loss_x_duration_hit for §4.3."
        ),
    }
    (out_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[{RUN_LABEL}] units={len(jobs)} scenarios={len(sc)} "
        f"n_sims={args.n_sims} workers={workers}"
    )
    print(f"  out -> {out_root}")
    print(f"  scenarios: {', '.join(s.key for s in sc)}")
    print(
        "  schedule (largest |V| first): "
        + ", ".join(f"{j['unit']}(n={j.get('n_nodes', '?')})" for j in jobs[:8])
        + (" ..." if len(jobs) > 8 else ""),
        flush=True,
    )

    # Cap workers at n_jobs (no benefit beyond one process per unit)
    workers = max(1, min(int(workers), len(jobs)))
    print(f"  workers (capped): {workers}", flush=True)

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
            args.epicenter_mode,
            args.network_year,
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
        f"[{RUN_LABEL}] done ok={ok} skip={skip} other={len(results)-ok-skip} "
        f"in {(time.time()-t0)/60:.1f} min"
    )
    other = len(results) - ok - skip
    if other:
        raise RuntimeError(f"Intervention run ended with {other} incomplete units")


if __name__ == "__main__":
    main()
