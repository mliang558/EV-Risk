#!/usr/bin/env python3
"""Require population-epicenter coverage for every modeled county."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from select_epicenter_by_population import require_pop_units


ROOT = Path(__file__).resolve().parents[2]


def fips5(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="raise").astype("int64")
    return numbers.astype(str).str.zfill(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population-path",
        type=Path,
        help="Validate a candidate GPKG/Parquet before replacing the active cache",
    )
    args = parser.parse_args()

    county_path = ROOT / "advi_results_2018_2023" / "county_estimates_advi.csv"
    counties = pd.read_csv(county_path)
    required = set(fips5(counties["fips"]))

    if args.population_path:
        import geopandas as gpd

        candidate = args.population_path.absolute()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        population = gpd.read_file(candidate)
        if population is None or population.empty:
            raise ValueError(f"No population rows in {candidate}")
    else:
        population = require_pop_units(ROOT).copy()
    population["county_fips"] = fips5(population["county_fips"])
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    present = set(population.loc[population["population"].gt(0), "county_fips"])

    missing = sorted(required - present)
    print(f"Population rows: {len(population):,}")
    print(f"Modeled counties: {len(required):,}")
    print(f"Covered modeled counties: {len(required) - len(missing):,}")
    if missing:
        print(f"Missing counties ({len(missing):,}): {', '.join(missing[:30])}")
        raise SystemExit("FAIL: population epicenters do not cover every modeled county")

    print("PASS: population epicenters cover every modeled county")


if __name__ == "__main__":
    main()
