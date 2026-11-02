#!/usr/bin/env python3
"""Validate a completed baseline plus Demand-comparator run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOSES = (0, 10, 20, 30, 50)
SCENARIOS = ["baseline", *(f"Demand_p{dose}" for dose in DOSES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sims", type=int, default=100)
    parser.add_argument("--expected-units", type=int, default=45)
    args = parser.parse_args()

    root = args.out.absolute()
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scenarios") != SCENARIOS:
        raise ValueError(f"Unexpected scenarios: {manifest.get('scenarios')}")
    if manifest.get("epicenter") != "county_uniform":
        raise ValueError("Demand run must use county-uniform outage epicenters")
    if not manifest.get("demand_road_source", {}).get("sha256"):
        raise ValueError("Manifest does not record the Demand road source")
    if not manifest.get("demand_population_source", {}).get("sha256"):
        raise ValueError("Manifest does not record the 2020 tract-population source")

    unit_dirs = sorted(path.parent for path in root.glob("*/summary.csv"))
    if len(unit_dirs) != args.expected_units:
        raise ValueError(f"Expected {args.expected_units} units, found {len(unit_dirs)}")

    for unit_dir in unit_dirs:
        summary = pd.read_csv(unit_dir / "summary.csv")
        if summary["scenario"].tolist() != SCENARIOS:
            raise ValueError(f"{unit_dir.name}: incomplete or misordered scenarios")
        if not np.isfinite(summary["L_original_mean"]).all():
            raise ValueError(f"{unit_dir.name}: non-finite original-node loss")

        metrics = {}
        for scenario in SCENARIOS:
            table = pd.read_csv(unit_dir / f"metrics_{scenario}.csv")
            if len(table) != args.expected_sims:
                raise ValueError(
                    f"{unit_dir.name}/{scenario}: expected {args.expected_sims} rows, "
                    f"found {len(table)}"
                )
            metrics[scenario] = table.sort_values("sim_id").reset_index(drop=True)
        if not np.allclose(
            metrics["baseline"]["L_original"],
            metrics["Demand_p0"]["L_original"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"{unit_dir.name}: Demand_p0 differs from baseline")

        metadata = json.loads((unit_dir / "densify_meta.json").read_text(encoding="utf-8"))
        for dose in DOSES[1:]:
            meta = metadata[f"Demand_p{dose}"]
            target, added = int(meta["n_target"]), int(meta["n_added"])
            if (
                meta.get("mode") != "population_primary_road_matched"
                or added != target
                or int(meta["n_shortfall"]) != 0
                or not meta.get("road_source", {}).get("sha256")
            ):
                raise ValueError(f"{unit_dir.name}/Demand_p{dose}: invalid placement metadata")
            distances = (
                meta.get("min_distance_to_existing_m"),
                meta.get("min_distance_between_added_m"),
            )
            if any(value is not None and float(value) < 10_000 - 1e-6 for value in distances):
                raise ValueError(f"{unit_dir.name}/Demand_p{dose}: spacing below 10 km")

    print(
        f"PASS: Demand comparator has {len(unit_dirs)} units, {args.expected_sims} "
        "replications, matched node counts, official road provenance, valid losses, "
        "county-uniform outage epicenters, complete population provenance, and 10 km spacing"
    )


if __name__ == "__main__":
    main()
