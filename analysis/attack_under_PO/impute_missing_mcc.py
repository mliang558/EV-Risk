#!/usr/bin/env python3
"""Fill missing legacy MCC counties without mixing customer-data vintages.

Existing MCC values are never changed.  For a missing county c in state s,

    ratio[s] = sum(existing MCC in s) / population covered by existing MCC
    MCC_hat[c] = population[c] * ratio[s].

Official 2020 Census county and state totals provide the population scale.
The script writes a two-column MCC file for the simulator and a separate audit
table describing every imputed county.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


STATE_FIPS = {"CA": "06", "NC": "37", "VA": "51", "WY": "56"}
EXPECTED_MISSING = {
    "06105",
    "37075",
    "37087",
    "37099",
    "37113",
    "37173",
    "37175",
    "51720",
    "56031",
}
DEFAULT_POPULATION_SUPPLEMENT = (
    Path(__file__).resolve().parent
    / "reference"
    / "missing_mcc_population_2020.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mcc(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"County_FIPS", "Customers"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")

    table = table[["County_FIPS", "Customers"]].copy()
    table["County_FIPS"] = pd.to_numeric(table["County_FIPS"], errors="coerce")
    table["Customers"] = pd.to_numeric(table["Customers"], errors="coerce")
    table = table.dropna(subset=["County_FIPS", "Customers"])
    table["County_FIPS"] = table["County_FIPS"].astype(int).astype(str).str.zfill(5)
    if table["County_FIPS"].duplicated().any():
        duplicates = table.loc[table["County_FIPS"].duplicated(), "County_FIPS"].tolist()
        raise ValueError(f"Duplicate MCC FIPS: {duplicates[:10]}")
    if (table["Customers"] <= 0).any():
        bad = table.loc[table["Customers"] <= 0, "County_FIPS"].tolist()
        raise ValueError(f"Non-positive existing MCC values: {bad[:10]}")
    return table


def _read_population_reference(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"county_fips", "population", "state_population"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    table = table.copy()
    table["county_fips"] = (
        pd.to_numeric(table["county_fips"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )
    for column in ("population", "state_population"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["population", "state_population"]].isna().any().any():
        raise ValueError("Population reference contains missing or invalid values")
    if (table[["population", "state_population"]] <= 0).any().any():
        raise ValueError("Population reference contains non-positive values")
    reference_fips = set(table["county_fips"])
    if reference_fips != EXPECTED_MISSING:
        raise ValueError(
            "Population reference must contain exactly the nine target counties. "
            f"Missing={sorted(EXPECTED_MISSING - reference_fips)}, "
            f"extra={sorted(reference_fips - EXPECTED_MISSING)}"
        )
    return table


def build_imputed_mcc(
    mcc: pd.DataFrame,
    population_reference: pd.DataFrame,
    states: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefixes = {STATE_FIPS[state] for state in states}
    target_fips = {
        fips for fips in EXPECTED_MISSING if fips[:2] in prefixes
    }
    imputation_population = population_reference[
        population_reference["county_fips"].isin(target_fips)
    ].copy()

    # These FIPS are known omissions in the official legacy MCC.  Exclude any
    # temporary values that may already have been copied into MCC.csv, then
    # reconstruct all nine by one consistent rule.
    existing_mcc = mcc[~mcc["County_FIPS"].isin(target_fips)].copy()
    calibration_mcc = existing_mcc.copy()
    calibration_mcc["state_fips"] = calibration_mcc["County_FIPS"].str[:2]
    calibration_mcc = calibration_mcc[
        calibration_mcc["state_fips"].isin(prefixes)
    ].copy()
    donor_count = calibration_mcc.groupby("state_fips").size().reindex(
        prefixes, fill_value=0
    )
    too_small = donor_count[donor_count < 10]
    if not too_small.empty:
        raise ValueError(f"Too few donor counties: {too_small.to_dict()}")

    imputation_population["state_fips"] = imputation_population["county_fips"].str[:2]
    state_reference = imputation_population.groupby("state_fips").agg(
        missing_population=("population", "sum"),
        state_population=("state_population", "first"),
        state_population_versions=("state_population", "nunique"),
    )
    if (state_reference["state_population_versions"] != 1).any():
        raise ValueError("A state has inconsistent population totals")
    state_reference["donor_population"] = (
        state_reference["state_population"]
        - state_reference["missing_population"]
    )
    state_reference["donor_customers"] = calibration_mcc.groupby("state_fips")[
        "Customers"
    ].sum()
    state_reference["state_ratio"] = (
        state_reference["donor_customers"]
        / state_reference["donor_population"]
    )
    state_ratio = state_reference["state_ratio"]
    imputation_population["state_ratio"] = imputation_population["state_fips"].map(
        state_ratio
    )
    invalid_ratio = ~imputation_population["state_ratio"].gt(0)
    if invalid_ratio.any():
        bad_states = sorted(imputation_population.loc[invalid_ratio, "state_fips"].unique())
        raise ValueError(f"Invalid state calibration ratios: {bad_states}")
    imputation_population["Customers"] = (
        imputation_population["population"] * imputation_population["state_ratio"]
    ).round().clip(lower=1).astype(int)
    reverse_state = {value: key for key, value in STATE_FIPS.items()}
    imputation_population["state"] = imputation_population["state_fips"].map(
        reverse_state
    )
    imputation_population["source"] = "state_aggregate_customers_per_population"
    imputation_population["population_source"] = (
        "2020_decennial_census_supplement"
    )
    previous = mcc.set_index("County_FIPS")["Customers"]
    imputation_population["previous_customers"] = imputation_population[
        "county_fips"
    ].map(previous)

    additions = imputation_population[["county_fips", "Customers"]].rename(
        columns={"county_fips": "County_FIPS"}
    )
    completed = pd.concat([existing_mcc, additions], ignore_index=True)
    completed = completed.sort_values("County_FIPS").reset_index(drop=True)
    completed["Customers"] = completed["Customers"].round().astype(int)

    original = existing_mcc.set_index("County_FIPS")["Customers"]
    preserved = completed.set_index("County_FIPS").loc[original.index, "Customers"]
    if not preserved.equals(original.astype(int)):
        raise AssertionError("An existing MCC value changed")
    if completed["County_FIPS"].duplicated().any():
        raise AssertionError("Completed MCC contains duplicate FIPS")
    expected_rows = len(existing_mcc) + len(target_fips)
    if len(completed) != expected_rows:
        raise AssertionError(
            f"Completed MCC has {len(completed)} rows; expected {expected_rows}"
        )

    audit = imputation_population[
        [
            "county_fips",
            "state",
            "population",
            "state_ratio",
            "previous_customers",
            "Customers",
            "source",
            "population_source",
        ]
    ].rename(columns={"county_fips": "County_FIPS"})
    return completed, audit.sort_values("County_FIPS").reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcc-in",
        type=Path,
        default=root / "notebooks" / "PO_data_cleaning" / "MCC_2022_original.csv",
    )
    parser.add_argument(
        "--population-supplement",
        type=Path,
        default=DEFAULT_POPULATION_SUPPLEMENT,
    )
    parser.add_argument(
        "--mcc-out",
        type=Path,
        default=root / "notebooks" / "PO_data_cleaning" / "MCC.csv",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=root / "results_review" / "mcc_population_imputation.csv",
    )
    parser.add_argument("--states", nargs="+", default=["CA", "NC", "VA", "WY"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    states = [state.upper() for state in args.states]
    unknown = sorted(set(states) - set(STATE_FIPS))
    if unknown:
        parser.error(f"Unsupported states: {unknown}")
    for path in (args.mcc_in, args.population_supplement):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.mcc_in.resolve() == args.mcc_out.resolve():
        raise ValueError("--mcc-in and --mcc-out must be different files")
    for path in (args.mcc_out, args.audit_out):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {path}; use --force")

    completed, audit = build_imputed_mcc(
        _read_mcc(args.mcc_in),
        _read_population_reference(args.population_supplement),
        states,
    )
    args.mcc_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(args.mcc_out, index=False)
    audit.to_csv(args.audit_out, index=False)

    metadata = {
        "method": "state aggregate of legacy MCC / 2020 Census population",
        "states": states,
        "source_mcc": str(args.mcc_in.resolve()),
        "source_mcc_sha256": _sha256(args.mcc_in),
        "output_mcc_sha256": _sha256(args.mcc_out),
        "population_supplement": str(args.population_supplement.resolve()),
        "population_supplement_sha256": _sha256(args.population_supplement),
        "n_existing_preserved": len(completed) - len(audit),
        "n_imputed": len(audit),
        "imputed_fips": audit["County_FIPS"].tolist(),
    }
    metadata_path = args.audit_out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(audit.to_string(index=False))
    print(f"PASS: preserved {metadata['n_existing_preserved']} MCC rows; imputed {len(audit)}")
    print(f"MCC:   {args.mcc_out}")
    print(f"Audit: {args.audit_out}")


if __name__ == "__main__":
    main()
