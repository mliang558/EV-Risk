#!/usr/bin/env python3
"""
Step 1: Export per-station charging data from AFDC historical CSV.

For each EV station (ELEC), keep latitude/longitude, L1/L2/DCFC counts, and:
    capacity = 5 * n_l1 + 25 * n_l2 + 300 * n_dc

Filters to the same 49 jurisdictions as run_charging_network.py
(48 contiguous states + DC; excludes AK, HI, territories).

Usage:
  python analysis/prepare_stations.py
  python analysis/prepare_stations.py --year 2026
  python analysis/prepare_stations.py --csv "data/raw/alt_fuel_stations_historical_day (Jan 1 2025).csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from run_charging_network import STATES_ORDER, load_raw_csv


OUTPUT_COLUMNS = [
    "state",
    "lat",
    "lon",
    "n_l1",
    "n_l2",
    "n_dc",
    "capacity",
]


def prepare_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to analysis states and export Step 1 columns."""
    sub = df[df["State"].astype(str).str.strip().isin(STATES_ORDER)].copy()
    l1 = pd.to_numeric(sub.get("EV Level1 EVSE Num", 0), errors="coerce").fillna(0)
    l2 = pd.to_numeric(sub.get("EV Level2 EVSE Num", 0), errors="coerce").fillna(0)
    dc = pd.to_numeric(sub.get("EV DC Fast Count", 0), errors="coerce").fillna(0)
    out = pd.DataFrame(
        {
            "state": sub["State"].astype(str).str.strip(),
            "lat": sub["lat"],
            "lon": sub["lon"],
            "n_l1": l1,
            "n_l2": l2,
            "n_dc": dc,
            "capacity": sub["capacity"],
        }
    )
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def default_csv_path(project_root: Path, year: int) -> Path:
    return project_root / "data" / "raw" / f"alt_fuel_stations_historical_day (Jan 1 {year}).csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Step 1 per-station AFDC table")
    parser.add_argument("--year", type=int, default=2026, help="Year in AFDC filename (default: 2026)")
    parser.add_argument("--csv", type=str, default=None, help="Override raw CSV path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: data/processed/stations_<year>_48states.csv)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else default_csv_path(project_root, args.year)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    m = re.search(r"Jan 1 (\d{4})\)", csv_path.name)
    year = int(m.group(1)) if m else args.year

    out_path = Path(args.out) if args.out else project_root / "data" / "processed" / f"stations_{year}_48states.csv"
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {csv_path}")
    df = load_raw_csv(csv_path)
    out = prepare_stations(df)
    out = out[OUTPUT_COLUMNS]
    out.to_csv(out_path, index=False)

    n_states = out["state"].nunique()
    print(f"Wrote {len(out):,} stations ({n_states} states/DC) -> {out_path}")
    print(f"  capacity range: [{out['capacity'].min():.0f}, {out['capacity'].max():.0f}]")


if __name__ == "__main__":
    main()
