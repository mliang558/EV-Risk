#!/usr/bin/env python3
"""Fail-fast checks for the canonical 2023 attack inputs."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ATTACK = ROOT / "analysis" / "attack_under_PO"
for path in (ROOT, ATTACK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_impact_radius import resolve_county_shapefile  # noqa: E402
from sample_lambda_from_posterior import (  # noqa: E402
    aggregate_state_lambda,
    load_posterior,
)


def require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"OK  {label}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network-root",
        default="outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--expected-units", type=int, default=45)
    args = parser.parse_args()

    network_root = Path(args.network_root)
    if not network_root.is_absolute():
        network_root = ROOT / network_root
    year_dir = network_root / str(args.year)

    summary_path = require(year_dir / "network_summary.csv", "network summary")
    summary = pd.read_csv(summary_path)
    if len(summary) != args.expected_units or summary["unit"].nunique() != args.expected_units:
        raise ValueError(
            f"Expected {args.expected_units} unique units, found "
            f"{len(summary)} rows / {summary['unit'].nunique()} unique units"
        )
    if (pd.to_numeric(summary["n_nodes"], errors="coerce") < 2).any():
        raise ValueError("Every canonical network must contain at least two nodes")

    net_dir = year_dir / "network_structures"
    for unit in summary["unit"].astype(str):
        pkl_path = net_dir / f"network_{unit}.pkl"
        if not pkl_path.is_file():
            pkl_path = net_dir / f"network_region_{unit}.pkl"
        require(pkl_path, f"network {unit}")
        with pkl_path.open("rb") as handle:
            graph = pickle.load(handle)
        if graph.number_of_nodes() < 2:
            raise ValueError(f"{unit}: graph has fewer than two nodes")

    require(
        ROOT / "advi_results_2018_2023" / "county_advi_posterior.nc",
        "county posterior",
    )
    county_csv = require(
        ROOT / "advi_results_2018_2023" / "county_estimates_advi.csv",
        "county estimates",
    )
    county_table = pd.read_csv(county_csv)
    needed = {"state", "fips"}
    if not needed.issubset(county_table.columns):
        raise ValueError(f"County estimates missing columns: {sorted(needed - set(county_table.columns))}")
    if county_table["fips"].duplicated().any():
        raise ValueError("County estimates contain duplicate FIPS rows")

    county_post, flat_lam = load_posterior(force_reload=True)
    if flat_lam.ndim != 2 or flat_lam.shape[1] != len(county_post):
        raise ValueError("Posterior county dimension does not match county estimates")
    if not np.isfinite(flat_lam).all() or (flat_lam < 0).any():
        raise ValueError("Posterior lambda samples must be finite and nonnegative")

    deterministic = county_post[["state", "fips"]].copy()
    deterministic["lambda_draw"] = flat_lam[0]
    state_rates = aggregate_state_lambda(deterministic)
    expected = deterministic.groupby("state", observed=True)["lambda_draw"].sum()
    observed = state_rates.set_index("state")["lambda_state_sum_draw"]
    if not np.allclose(observed.sort_index(), expected.sort_index()):
        raise AssertionError("State lambda is not the sum of county lambdas")
    print(
        "OK  state rate = county-rate sum; "
        f"range {observed.min():.2f}--{observed.max():.2f} county-events/year"
    )

    outages_path = require(
        ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv",
        "cleaned outage events",
    )
    outages = pd.read_csv(outages_path, low_memory=False)
    if "year" not in outages.columns:
        if "start_time" not in outages.columns:
            raise ValueError("Outage events need either year or start_time")
        years = pd.to_datetime(outages["start_time"], errors="coerce").dt.year
    else:
        years = pd.to_numeric(outages["year"], errors="coerce")
    missing_years = sorted(set(range(2018, 2024)) - set(years.dropna().astype(int)))
    if missing_years:
        raise ValueError(f"Cleaned outage event pool is missing years: {missing_years}")
    if not ({"mean_customers", "affected_customers"} & set(outages.columns)):
        raise ValueError("Outage events need mean_customers or affected_customers")

    require(ROOT / "notebooks" / "PO_data_cleaning" / "MCC.csv", "county customer counts")
    require(ROOT / "data" / "processed" / "pop_units_epicenter.gpkg", "population epicenters")
    require(ROOT.parent / "24237376" / "coverage_history.csv", "coverage history")
    print(f"OK  county polygons: {resolve_county_shapefile(ROOT)}")
    print("PASS: canonical inputs and state-rate construction are consistent")


if __name__ == "__main__":
    main()
