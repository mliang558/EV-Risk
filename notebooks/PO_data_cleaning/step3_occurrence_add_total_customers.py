"""
Step 3 (compact): attach county-level total customers (MCC.csv) to cleaned outages with population.

Inputs:
- cleaned outages with (optionally) population already merged, e.g.
  - outages_with_population_2018_2023.csv   (default), or
  - cleaned_outages_2018_2023.csv           (if you run without step 2)

- MCC.csv with total customers per county:
    columns: ['County_FIPS', 'Customers']

Logic:
- Join on FIPS: outages['fips'] == MCC['County_FIPS']
- Add a column 'total_customers' (renamed from MCC.Customers)
- Optionally add '%_mean_customers_total' = mean_customers / total_customers

No plotting, non-interactive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_total_customers(
    outages: pd.DataFrame,
    mcc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge MCC total customers into outages on county FIPS.
    """
    out = outages.copy()
    mcc_df = mcc.copy()

    # Normalize FIPS dtype for safe join
    if "County_FIPS" not in mcc_df.columns or "Customers" not in mcc_df.columns:
        raise ValueError("MCC.csv must contain columns 'County_FIPS' and 'Customers'.")

    # Some MCC files contain total / header rows like "Grand Total" in County_FIPS.
    # Coerce to numeric, drop non-county rows, then cast to int.
    mcc_df["County_FIPS"] = pd.to_numeric(mcc_df["County_FIPS"], errors="coerce")
    mcc_df = mcc_df.dropna(subset=["County_FIPS"])
    mcc_df["County_FIPS"] = mcc_df["County_FIPS"].astype(int)

    # Step1 uses integer-like FIPS (e.g. 1001)
    out["fips"] = out["fips"].astype(int)

    mcc_df = mcc_df.rename(columns={"Customers": "total_customers"})

    merged = pd.merge(
        out,
        mcc_df[["County_FIPS", "total_customers"]],
        left_on="fips",
        right_on="County_FIPS",
        how="left",
    )

    # Drop join helper if not needed
    merged = merged.drop(columns=["County_FIPS"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=Path(__file__).resolve().parent / "outages_with_population_2018_2023.csv",
        help="Cleaned outages CSV from step 2 (default: outages_with_population_2018_2023.csv in this folder). "
             "If you skip step 2, point this to cleaned_outages_2018_2023.csv instead.",
    )
    parser.add_argument(
        "--mcc-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "MCC.csv",
        help="MCC county total customers CSV (default: MCC.csv in this folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outages_with_population_and_MCC_2018_2023.csv",
        help="Output CSV path (default: outages_with_population_and_MCC_2018_2023.csv in this folder).",
    )
    parser.add_argument(
        "--add-customers-pct-total",
        action="store_true",
        help="If set, add '%_mean_customers_total' = mean_customers / total_customers when both exist.",
    )
    args = parser.parse_args()

    if not args.cleaned_input.exists():
        raise FileNotFoundError(f"Cleaned input file not found: {args.cleaned_input}")
    if not args.mcc_csv.exists():
        raise FileNotFoundError(f"MCC CSV not found: {args.mcc_csv}")

    outages = pd.read_csv(args.cleaned_input)
    mcc = pd.read_csv(args.mcc_csv)

    merged = add_total_customers(outages, mcc)

    # Simple coverage check: how many outages successfully matched an MCC total?
    if "total_customers" in merged.columns:
        total_rows = len(merged)
        matched_rows = merged["total_customers"].notna().sum()
        missing_rows = total_rows - matched_rows
        pct = matched_rows / total_rows if total_rows else 0.0
        print(
            f"MCC join coverage: {matched_rows}/{total_rows} rows matched "
            f"({pct:.1%}), {missing_rows} without total_customers."
        )
        if missing_rows > 0:
            sample_missing = (
                merged[merged["total_customers"].isna()][["fips", "state", "county"]]
                .drop_duplicates()
                .head(10)
            )
            print("Example FIPS/state/county with missing MCC total (up to 10):")
            print(sample_missing.to_string(index=False))

    if args.add_customers_pct_total and "total_customers" in merged.columns and "mean_customers" in merged.columns:
        merged["%_mean_customers_total"] = merged["mean_customers"] / merged["total_customers"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  rows={len(merged):,}")


if __name__ == "__main__":
    main()

