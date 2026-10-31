#!/usr/bin/env python3
"""Validate one small intervention run before launching all 45 units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "analysis" / "attack_under_PO").is_dir():
    ROOT = Path(__file__).absolute().parents[3]
DOSES = (0, 10, 20, 30, 50)


class Checks:
    def __init__(self) -> None:
        self.failures = []

    def check(self, condition: bool, message: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            self.failures.append(message)


def _events(unit_dir: Path, scenario: str) -> pd.DataFrame:
    path = unit_dir / f"events_{scenario}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; smoke runs must retain events")
    return pd.read_csv(path).sort_values(["sim_id", "event_id"]).reset_index(drop=True)


def _metrics(unit_dir: Path, scenario: str) -> pd.DataFrame:
    path = unit_dir / f"metrics_{scenario}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path).sort_values("sim_id").reset_index(drop=True)


def _check_common_draws(unit_dir: Path, checks: Checks) -> None:
    baseline = _events(unit_dir, "baseline")
    columns = ["sim_id", "event_id", "epicenter_lat", "epicenter_lon"]
    base_duration = baseline["duration_hours"].to_numpy()
    for family in ("CF-D_nevi_p", "CF-S_m"):
        for dose in DOSES:
            scenario = f"{family}{dose}"
            events = _events(unit_dir, scenario)
            same_rows = len(events) == len(baseline)
            same_draw = same_rows and events[columns].equals(baseline[columns])
            duration = events.get("duration_base_hours", events["duration_hours"])
            same_duration = same_rows and np.allclose(duration, base_duration)
            checks.check(same_draw and same_duration, f"{scenario}: shared events and epicenters")

    baseline_loss = _metrics(unit_dir, "baseline")["L_tilde"].to_numpy()
    for scenario in ("CF-D_nevi_p0", "CF-S_m0"):
        loss = _metrics(unit_dir, scenario)["L_tilde"].to_numpy()
        checks.check(
            np.allclose(loss, baseline_loss, atol=1e-12, rtol=0),
            f"{scenario}: dose zero equals baseline",
        )


def _check_dose_response(unit_dir: Path, checks: Checks) -> None:
    severity_loss = [
        _metrics(unit_dir, f"CF-S_m{dose}")["L_tilde"].mean() for dose in DOSES
    ]
    checks.check(
        all(a + 1e-12 >= b for a, b in zip(severity_loss, severity_loss[1:])),
        "CF-S loss does not increase as severity is reduced",
    )

    hit_tables = []
    for dose in DOSES:
        table = _events(unit_dir, f"CF-D_nevi_p{dose}")
        hit_tables.append(table[["sim_id", "event_id", "hit"]].rename(columns={"hit": dose}))
    hits = hit_tables[0]
    for table in hit_tables[1:]:
        hits = hits.merge(table, on=["sim_id", "event_id"], validate="one_to_one")
    checks.check(
        all((hits[b] >= hits[a]).all() for a, b in zip(DOSES, DOSES[1:])),
        "CF-D event hits do not decrease when nodes are added",
    )

    baseline = _metrics(unit_dir, "baseline")
    checks.check(
        baseline["P_hit"].between(0, 1).all(),
        "baseline hit probabilities are valid",
    )


def _check_corridor_metadata(unit_dir: Path, checks: Checks) -> None:
    path = unit_dir / "densify_meta.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for dose in DOSES[1:]:
        key = f"CF-D_nevi_p{dose}"
        meta = metadata[key]
        target, added = int(meta["n_target"]), int(meta["n_added"])
        checks.check(
            0 <= added <= target and int(meta["n_shortfall"]) == target - added,
            f"{key}: additions and corridor shortfall are internally consistent",
        )
        checks.check(
            meta.get("mode") == "fhwa_ev_afc"
            and np.isclose(float(meta.get("spacing_m", 0)), 50 * 1609.344)
            and np.isclose(float(meta.get("cluster_radius_m", 0)), 10_000),
            f"{key}: official EV AFC placement rules are recorded",
        )
        checks.check(
            int(meta.get("n_candidates", 0))
            >= int(meta.get("n_candidates_eligible", 0))
            >= added,
            f"{key}: corridor candidate counts are valid",
        )
        distances = [
            meta.get("min_distance_to_existing_m"),
            meta.get("min_distance_between_added_m"),
        ]
        checks.check(
            all(value is None or float(value) >= 10_000 - 1e-6 for value in distances),
            f"{key}: added sites respect the 10 km separation rule",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()

    batch = args.batch_dir if args.batch_dir.is_absolute() else ROOT / args.batch_dir
    unit_dir = batch / args.unit
    if not unit_dir.is_dir():
        raise FileNotFoundError(unit_dir)

    checks = Checks()
    _check_common_draws(unit_dir, checks)
    _check_dose_response(unit_dir, checks)
    _check_corridor_metadata(unit_dir, checks)
    if checks.failures:
        print(f"FAIL: {len(checks.failures)} checks failed")
        sys.exit(1)
    print("PASS: intervention smoke test")


if __name__ == "__main__":
    main()
