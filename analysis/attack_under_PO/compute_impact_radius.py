#!/usr/bin/env python3
"""
Compute impact radius r for outage events using:

    r = sqrt( min(N_d / theta_s, MCC_c*) / (rho_c* * pi) )

where:
- N_d       : impacted customers for this event
- theta_s   : state-level Eagle-I coverage (per year)
- MCC_c*    : total customers in county c* (from MCC.csv)
- rho_c*    : customer density = MCC_c* / A_c* (A_c* = county area in km^2)

Inputs (paths are relative to Pro_directory):
- advi_results_2018_2023/bootstrapped_outages_sample1.csv
- 24237376/coverage_history.csv
- notebooks/PO_data_cleaning/MCC.csv
- tl_2021_us_county/tl_2021_us_county.shp

Output:
- advi_results_2018_2023/bootstrapped_outages_with_radius_sample1.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import geopandas as gpd


def resolve_county_shapefile(project_root: Path) -> Path:
    """Find tl_2021_us_county.shp (repo parent, or attack_under_PO/demographic data)."""
    here = Path(__file__).resolve().parent
    candidates = [
        project_root.parent / "tl_2021_us_county" / "tl_2021_us_county.shp",
        project_root / "tl_2021_us_county" / "tl_2021_us_county.shp",
        here / "demographic data" / "tl_2021_us_county.shp",
        here / "module_3_03" / "demographic data" / "tl_2021_us_county.shp",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "County shapefile not found. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


STATE_NAME_TO_ABBR: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def load_county_geometry_and_mcc(project_root: Path) -> pd.DataFrame:
    """
    Load county shapefile and MCC, compute area and density.

    Returns DataFrame with:
    - fips_str
    - MCC (total customers)
    - area_km2
    - rho (customers / km^2)
    """
    shp_path = resolve_county_shapefile(project_root)
    mcc_path = project_root / "notebooks" / "PO_data_cleaning" / "MCC.csv"
    if not mcc_path.exists():
        raise FileNotFoundError(f"MCC.csv not found: {mcc_path}")

    gdf = gpd.read_file(shp_path)

    # Only CONUS + DC
    if "STATEFP" in gdf.columns:
        non_continental = {"02", "15", "72", "78", "60", "66", "69"}
        gdf = gdf[~gdf["STATEFP"].astype(str).isin(non_continental)].copy()

    if "GEOID" not in gdf.columns:
        raise ValueError("Shapefile must contain GEOID column for county FIPS.")

    gdf["fips_str"] = gdf["GEOID"].astype(str).str.zfill(5)

    # Project to equal-area CRS for proper area calculation (US national Albers)
    gdf_proj = gdf.to_crs("EPSG:5070")
    gdf["area_km2"] = gdf_proj.geometry.area.values / 1e6

    mcc = pd.read_csv(mcc_path)
    if not {"County_FIPS", "Customers"}.issubset(mcc.columns):
        raise ValueError("MCC.csv must contain 'County_FIPS' and 'Customers'.")

    # 清理非县级汇总行（例如 'Grand Total'）
    mcc["County_FIPS"] = pd.to_numeric(mcc["County_FIPS"], errors="coerce")
    mcc = mcc.dropna(subset=["County_FIPS"])

    mcc["fips_str"] = mcc["County_FIPS"].astype(int).astype(str).str.zfill(5)
    mcc = mcc.rename(columns={"Customers": "MCC"})

    df = gdf[["fips_str", "area_km2"]].merge(
        mcc[["fips_str", "MCC"]], on="fips_str", how="left"
    )

    # Some counties may lack MCC; avoid division by zero
    df["MCC"] = df["MCC"].fillna(0.0)
    df["area_km2"] = df["area_km2"].clip(lower=1e-3)

    df["rho"] = df["MCC"] / df["area_km2"]
    return df


def load_coverage_history(project_root: Path) -> pd.DataFrame:
    """
    Load coverage_history.csv and extract theta_s (max_pct_covered) by state-year.
    """
    cov_path = project_root.parent / "24237376" / "coverage_history.csv"
    if not cov_path.exists():
        raise FileNotFoundError(f"coverage_history.csv not found: {cov_path}")

    cov = pd.read_csv(cov_path)
    # Parse year column (e.g., '1/1/18') to integer year with explicit format
    cov["year_num"] = pd.to_datetime(
        cov["year"], format="%m/%d/%y", errors="coerce"
    ).dt.year
    cov = cov.rename(columns={"state": "state_abbr"})

    # Use max_pct_covered as theta_s (coverage rate between 0–1)
    cov["theta_s"] = cov["max_pct_covered"].astype(float)
    return cov[["state_abbr", "year_num", "theta_s"]]


def compute_radius_for_events(project_root: Path) -> pd.DataFrame:
    """
    Load bootstrapped outages and append impact radius r (km) per event.
    """
    results_dir = project_root / "advi_results_2018_2023"
    boot_path = results_dir / "bootstrapped_outages_sample1.csv"
    if not boot_path.exists():
        raise FileNotFoundError(f"Bootstrapped outages CSV not found: {boot_path}")

    events = pd.read_csv(boot_path)

    # Expect year and state columns in events
    if "year" not in events.columns:
        # Try to infer from start_time if exists
        if "start_time" in events.columns:
            events["year"] = pd.to_datetime(events["start_time"]).dt.year
        else:
            raise ValueError("Events CSV must contain 'year' or 'start_time' column.")

    if "state" not in events.columns:
        raise ValueError("Events CSV must contain 'state' column (state names).")

    # Map state name -> abbreviation to match coverage_history
    events["state_abbr"] = events["state"].map(STATE_NAME_TO_ABBR)

    # FIPS as 5-digit string
    if "fips" not in events.columns:
        raise ValueError("Events CSV must contain 'fips' column.")
    events["fips_str"] = events["fips"].astype(int).astype(str).str.zfill(5)

    # Impacted customers N_d: prefer 'affected_customers', otherwise 'mean_customers'
    if "affected_customers" in events.columns:
        events["Nd"] = events["affected_customers"].astype(float)
    elif "mean_customers" in events.columns:
        events["Nd"] = events["mean_customers"].astype(float)
    else:
        raise ValueError(
            "Events CSV must contain 'affected_customers' or 'mean_customers' column."
        )

    # Load county MCC & area
    county_df = load_county_geometry_and_mcc(project_root)

    # Load coverage history theta_s
    cov_df = load_coverage_history(project_root)

    # Merge county info
    events = events.merge(county_df, on="fips_str", how="left")

    # Merge coverage info
    events = events.merge(
        cov_df,
        left_on=["state_abbr", "year"],
        right_on=["state_abbr", "year_num"],
        how="left",
    )

    # Clean up
    events["theta_s"] = events["theta_s"].fillna(1.0)  # if missing, assume full coverage
    events["theta_s"] = events["theta_s"].clip(lower=1e-3, upper=1.0)

    # Compute effective impacted customers per formula: min(Nd/theta_s, MCC_c*)
    Nd_eff = events["Nd"] / events["theta_s"]
    Nd_eff = Nd_eff.clip(lower=0.0)
    Nd_eff = np.minimum(Nd_eff, events["MCC"])

    # rho = MCC / area_km2 (already computed)
    rho = events["rho"].clip(lower=1e-6)

    # r = sqrt( Nd_eff / (rho * pi) ), in km
    events["impact_radius_km"] = np.sqrt(Nd_eff / (rho * np.pi))

    return events


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../Pro_directory

    events_with_r = compute_radius_for_events(project_root)

    results_dir = project_root / "advi_results_2018_2023"
    out_path = results_dir / "bootstrapped_outages_with_radius_sample1.csv"
    events_with_r.to_csv(out_path, index=False)

    print(f"✓ Saved outages with impact radius to: {out_path}")
    print(
        events_with_r[
            ["state", "fips_str", "year", "Nd", "theta_s", "MCC", "area_km2", "impact_radius_km"]
        ].head()
    )


if __name__ == "__main__":
    main()

