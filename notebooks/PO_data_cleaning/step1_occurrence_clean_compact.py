"""
Compact outage occurrence cleaning (extract from Step1-Occurrence.ipynb).

Goal:
- Load EagleI merged outage CSVs (2018-2023)
- Basic preprocessing (timestamps, duration/end_time, drop duplicates)
- Remove non-continental regions
- Merge (1) overlapping outages per county (data_clean1)
- Merge (2) close outages per county within a max gap (default 60 min) (data_clean2)
- Export cleaned CSV (default: cleaned_outages_2018_2023.csv)

This script intentionally excludes plotting / visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


STATES_TO_REMOVE = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "US Virgin Islands",
    "United States Virgin Islands",
}


def load_eaglei_merged_csvs(input_dir: Path, years: Iterable[int]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for y in years:
        p = input_dir / f"eaglei_outages_{y}_merged.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, axis=0, ignore_index=True)


def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the notebook's early preprocessing:
    - start_time -> datetime
    - duration_min = duration * 60
    - duration -> timedelta(minutes=duration_min)
    - end_time = start_time + duration
    - drop exact duplicate rows
    """
    out = df.copy()

    # Required columns (minimal sanity checks)
    required = {"fips", "state", "county", "start_time", "duration", "min_customers", "max_customers", "mean_customers"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    out["start_time"] = pd.to_datetime(out["start_time"])

    # Notebook assumed "duration" is in hours
    out["duration_min"] = out["duration"] * 60
    out["duration"] = pd.to_timedelta(out["duration_min"], unit="m")
    out["end_time"] = out["start_time"] + out["duration"]

    out = out.drop_duplicates()
    return out


def merge_outages(
    df: pd.DataFrame,
    *,
    max_gap_minutes: float = 0,
    mean_weight_col: str = "duration_min",
) -> pd.DataFrame:
    """
    Merge outages within the same FIPS if:
    - overlapping/touching (next_start <= curr_end), OR
    - time gap between curr_end and next_start <= max_gap_minutes

    Aggregation (consistent intent with notebook):
    - start_time: earliest
    - end_time: latest
    - min_customers: min
    - max_customers: max
    - mean_customers: duration-weighted mean using `mean_weight_col` (minutes)

    Keeps 'state' and 'county' from the first row per FIPS group.
    """
    out = df.copy()
    out["start_time"] = pd.to_datetime(out["start_time"])
    out["end_time"] = pd.to_datetime(out["end_time"])

    if mean_weight_col not in out.columns:
        raise ValueError(f"Missing weight column for mean aggregation: {mean_weight_col}")

    merged_rows: list[dict] = []

    for fips_code, group in out.groupby("fips"):
        group = group.sort_values("start_time").reset_index(drop=True)
        if group.empty:
            continue

        curr_state = group.loc[0, "state"]
        curr_county = group.loc[0, "county"]

        curr_start = group.loc[0, "start_time"]
        curr_end = group.loc[0, "end_time"]
        curr_min = group.loc[0, "min_customers"]
        curr_max = group.loc[0, "max_customers"]

        # maintain weighted mean = sum(mean * weight) / sum(weight)
        curr_weight = float(group.loc[0, mean_weight_col])
        curr_mean_weighted_sum = float(group.loc[0, "mean_customers"]) * curr_weight

        for i in range(1, len(group)):
            next_start = group.loc[i, "start_time"]
            next_end = group.loc[i, "end_time"]
            next_min = group.loc[i, "min_customers"]
            next_max = group.loc[i, "max_customers"]
            next_weight = float(group.loc[i, mean_weight_col])
            next_mean = float(group.loc[i, "mean_customers"])

            gap_minutes = (next_start - curr_end).total_seconds() / 60.0
            overlaps_or_close = (next_start <= curr_end) or (gap_minutes <= max_gap_minutes)

            if overlaps_or_close:
                curr_end = max(curr_end, next_end)
                curr_min = min(curr_min, next_min)
                curr_max = max(curr_max, next_max)
                curr_mean_weighted_sum += next_mean * next_weight
                curr_weight += next_weight
            else:
                merged_rows.append(
                    {
                        "fips": fips_code,
                        "state": curr_state,
                        "county": curr_county,
                        "start_time": curr_start,
                        "end_time": curr_end,
                        "min_customers": curr_min,
                        "max_customers": curr_max,
                        "mean_customers": curr_mean_weighted_sum / curr_weight if curr_weight else None,
                    }
                )
                curr_start = next_start
                curr_end = next_end
                curr_min = next_min
                curr_max = next_max
                curr_weight = next_weight
                curr_mean_weighted_sum = next_mean * next_weight

        merged_rows.append(
            {
                "fips": fips_code,
                "state": curr_state,
                "county": curr_county,
                "start_time": curr_start,
                "end_time": curr_end,
                "min_customers": curr_min,
                "max_customers": curr_max,
                "mean_customers": curr_mean_weighted_sum / curr_weight if curr_weight else None,
            }
        )

    return pd.DataFrame(merged_rows)[
        ["fips", "state", "county", "start_time", "end_time", "min_customers", "max_customers", "mean_customers"]
    ]


def add_duration_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["start_time"] = pd.to_datetime(out["start_time"])
    out["end_time"] = pd.to_datetime(out["end_time"])
    out["duration"] = out["end_time"] - out["start_time"]
    out["duration_min"] = out["duration"].dt.total_seconds() / 60.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing eaglei_outages_YYYY_merged.csv files (default: this script folder)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020, 2021, 2022, 2023],
        help="Years to include (default: 2018..2023)",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=60.0,
        help="Max gap minutes for merging close outages (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "cleaned_outages_2018_2023.csv",
        help="Output CSV path (default: cleaned_outages_2018_2023.csv in this folder)",
    )
    args = parser.parse_args()

    merged_df = load_eaglei_merged_csvs(args.input_dir, args.years)
    merged_df = preprocess_base(merged_df)

    # remove non-continental regions
    continental_df = merged_df[~merged_df["state"].isin(STATES_TO_REMOVE)].copy()

    # data_clean1: merge overlapping/touching events
    data_clean1 = merge_outages(continental_df, max_gap_minutes=0, mean_weight_col="duration_min")
    data_clean1["date"] = pd.to_datetime(data_clean1["start_time"]).dt.date
    data_clean1["time"] = pd.to_datetime(data_clean1["start_time"]).dt.time
    data_clean1 = add_duration_cols(data_clean1)

    # data_clean2: merge close events (overlap or gap <= max_gap)
    data_clean2 = merge_outages(data_clean1, max_gap_minutes=args.max_gap_minutes, mean_weight_col="duration_min")
    data_clean2["date"] = pd.to_datetime(data_clean2["start_time"]).dt.date
    data_clean2["time"] = pd.to_datetime(data_clean2["start_time"]).dt.time
    data_clean2 = add_duration_cols(data_clean2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_clean2.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  rows={len(data_clean2):,}")


if __name__ == "__main__":
    main()

