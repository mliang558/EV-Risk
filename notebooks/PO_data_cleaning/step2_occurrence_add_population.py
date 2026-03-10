"""
Step 2 (compact): attach county population to cleaned outage data.

Input (from Step 1 script):
- cleaned outages file with columns at least:
  ['fips', 'state', 'county', 'start_time', 'end_time',
   'min_customers', 'max_customers', 'mean_customers',
   'date', 'time', 'duration', 'duration_min']

Population input (ACS 2022, same as notebook):
- acs_by_county.csv (in this folder), with at least:
  ['STATE', 'COUNTY', ..., 'POPESTIMATE2022', ...]

Logic (from Step1-Occurrence.ipynb):
- String-normalize state / county names in both tables
- Left-merge on (state_clean, county_clean)
- Drop helper columns and obvious unused ACS fields
- Optionally compute customers percentage if requested

This script is intentionally non-interactive and contains no plotting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_str(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = s.strip()
    s = re.sub(r"\\s+", " ", s)
    s = s.lower()
    return s


def add_population(
    outages: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach population columns to outages using cleaned (state, county) names.

    This mirrors the notebook's:
    - df1 = data_clean2
    - df2 = population
    - clean_str on state/county
    - merge on cleaned keys
    - drop unnamed + some ACS columns
    """
    df1 = outages.copy()
    df2 = population.copy()

    # Clean join keys on outages
    df1["state_clean"] = df1["state"].apply(clean_str)
    df1["county_clean"] = df1["county"].apply(clean_str)

    # Clean join keys on population
    if "COUNTY" not in df2.columns or "STATE" not in df2.columns:
        raise ValueError("Population CSV must contain columns 'STATE' and 'COUNTY'.")

    df2["COUNTY_clean"] = df2["COUNTY"].apply(clean_str)
    df2["STATE_clean"] = df2["STATE"].apply(clean_str)

    merged = pd.merge(
        df1,
        df2,
        left_on=["state_clean", "county_clean"],
        right_on=["STATE_clean", "COUNTY_clean"],
        how="left",
    )

    # Drop obvious helper / unused columns if present
    drop_cols = [
        "state_clean",
        "county_clean",
        "STATE_clean",
        "COUNTY_clean",
        "COUNTY",
        "COUNTYTYPE",
        "STATE",
    ]
    drop_cols += [c for c in merged.columns if c.startswith("Unnamed:")]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleaned-occurrence",
        type=Path,
        default=Path(__file__).resolve().parent / "cleaned_outages_2018_2023.csv",
        help="Input cleaned outages CSV from Step 1 (default: cleaned_outages_2018_2023.csv in this folder)",
    )
    parser.add_argument(
        "--population-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "acs_by_county.csv",
        help="ACS county population CSV (default: acs_by_county.csv in this folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outages_with_population_2018_2023.csv",
        help="Output CSV path (default: outages_with_population_2018_2023.csv in this folder)",
    )
    parser.add_argument(
        "--add-customers-pct",
        action="store_true",
        help="If set, add '%_mean_customers' = mean_customers / POPESTIMATE2022 when that column exists.",
    )
    args = parser.parse_args()

    if not args.cleaned_occurrence.exists():
        raise FileNotFoundError(f"Cleaned occurrence file not found: {args.cleaned_occurrence}")
    if not args.population_csv.exists():
        raise FileNotFoundError(f"Population CSV not found: {args.population_csv}")

    outages = pd.read_csv(args.cleaned_occurrence)
    population = pd.read_csv(args.population_csv)

    merged = add_population(outages, population)

    # Simple coverage check: how many outages successfully matched a population record?
    if "POPESTIMATE2022" in merged.columns:
        total_rows = len(merged)
        matched_rows = merged["POPESTIMATE2022"].notna().sum()
        missing_rows = total_rows - matched_rows
        pct = matched_rows / total_rows if total_rows else 0.0
        print(
            f"Population join coverage: {matched_rows}/{total_rows} rows matched "
            f"({pct:.1%}), {missing_rows} without population."
        )
        if missing_rows > 0:
            sample_missing = (
                merged[merged["POPESTIMATE2022"].isna()][["state", "county"]]
                .drop_duplicates()
                .head(10)
            )
            print("Example state/county with missing population (up to 10):")
            print(sample_missing.to_string(index=False))

    if args.add_customers_pct and "POPESTIMATE2022" in merged.columns and "mean_customers" in merged.columns:
        merged["%_mean_customers"] = merged["mean_customers"] / merged["POPESTIMATE2022"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  rows={len(merged):,}")


if __name__ == "__main__":
    main()

