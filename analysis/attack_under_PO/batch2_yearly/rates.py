"""Year-specific event counts N[s,t] and rate multipliers r[s,t]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import YEARS


def ensure_outage_year(outages: pd.DataFrame) -> pd.DataFrame:
    df = outages.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["start_time"]).dt.year
    df["year"] = df["year"].astype(int)
    if "fips" in df.columns and "fips_str" not in df.columns:
        df["fips_str"] = df["fips"].astype(int).astype(str).str.zfill(5)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d")
    return df


def state_year_counts(outages: pd.DataFrame, years: tuple[int, ...] = YEARS) -> pd.DataFrame:
    """N[s,t] = merged event count by state and start-year."""
    df = ensure_outage_year(outages)
    df = df[df["year"].isin(years)]
    ct = (
        df.groupby(["state", "year"], observed=True)
        .size()
        .rename("N")
        .reset_index()
    )
    # full grid
    states = sorted(df["state"].dropna().unique())
    grid = pd.MultiIndex.from_product([states, list(years)], names=["state", "year"])
    ct = ct.set_index(["state", "year"]).reindex(grid, fill_value=0).reset_index()
    return ct


def rate_multipliers(counts: pd.DataFrame) -> pd.DataFrame:
    """
    r[s,t] = N[s,t] / mean_t(N[s,t]).
    Assert mean_t r == 1 for every state with positive mean N.
    """
    df = counts.copy()
    means = df.groupby("state", observed=True)["N"].transform("mean")
    df["r"] = np.where(means > 0, df["N"] / means, 1.0)
    # states with all-zero years: r = 1
    chk = df.groupby("state", observed=True)["r"].mean()
    bad = chk[(chk - 1.0).abs() > 1e-8]
    if len(bad):
        raise AssertionError(f"mean_t r[s,t] != 1 for states: {bad.to_dict()}")
    return df


def build_r_table(outages: pd.DataFrame, years: tuple[int, ...] = YEARS) -> pd.DataFrame:
    return rate_multipliers(state_year_counts(outages, years))


def r_lookup(r_table: pd.DataFrame, state: str, year: int) -> float:
    hit = r_table[(r_table["state"] == state) & (r_table["year"] == int(year))]
    if hit.empty:
        return 1.0
    return float(hit["r"].iloc[0])


def save_r_table(r_table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r_table.to_csv(path, index=False)
