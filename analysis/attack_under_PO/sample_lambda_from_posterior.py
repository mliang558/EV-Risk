#!/usr/bin/env python3
"""
Sample county-level outage rates λ from the ADVI posterior.

This script does NOT depend on any pickle/network files.
It only uses:
- advi_results_2018_2023/county_advi_posterior.nc
- advi_results_2018_2023/county_estimates_advi.csv

Main uses:
- Draw one realization of λ_c for all counties (for a single Monte Carlo run)
- Optionally save that realization to CSV for downstream attack simulations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd


def load_posterior() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load county-level λ posterior samples from ADVI results.

    Returns
    -------
    county_df : DataFrame
        Must contain at least ['state', 'fips'] in the same order as the posterior.
    flat_lam : np.ndarray of shape (n_samples, n_counties)
        Flattened posterior samples of lambda_county across chains and draws.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../Pro_directory

    results_dir = project_root / "advi_results_2018_2023"
    posterior_path = results_dir / "county_advi_posterior.nc"
    county_csv = results_dir / "county_estimates_advi.csv"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior file not found: {posterior_path}")
    if not county_csv.exists():
        raise FileNotFoundError(f"County estimates CSV not found: {county_csv}")

    county_df = pd.read_csv(county_csv)
    if "fips" not in county_df.columns:
        raise ValueError("county_estimates_advi.csv must contain a 'fips' column.")

    print(f"Loading posterior from: {posterior_path}")
    idata = az.from_netcdf(posterior_path)

    lam = idata.posterior["lambda_county"].values  # (chains, draws, n_counties)
    flat_lam = lam.reshape(-1, lam.shape[-1])  # (n_samples, n_counties)

    print(f"Posterior shape: chains={lam.shape[0]}, draws={lam.shape[1]}, counties={lam.shape[2]}")
    print(f"Flattened samples: {flat_lam.shape[0]} x {flat_lam.shape[1]}")

    return county_df, flat_lam


def load_county_estimates() -> pd.DataFrame:
    """County table with lambda_raw and lambda_mean (posterior mean)."""
    project_root = Path(__file__).resolve().parents[2]
    county_csv = project_root / "advi_results_2018_2023" / "county_estimates_advi.csv"
    if not county_csv.is_file():
        raise FileNotFoundError(f"County estimates CSV not found: {county_csv}")
    return pd.read_csv(county_csv)


def build_raw_lambda_draw(county_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed county λ from empirical event rates (lambda_raw = n_events / n_years).

    Same schema as draw_lambda_realization output for downstream bootstrap.
    """
    if "lambda_raw" not in county_df.columns:
        raise ValueError("county_df must contain 'lambda_raw'")
    df_draw = county_df.copy()
    df_draw["lambda_draw"] = pd.to_numeric(df_draw["lambda_raw"], errors="coerce").fillna(0.0)
    df_draw["lambda_draw"] = df_draw["lambda_draw"].clip(lower=1e-8)
    return df_draw


def draw_lambda_realization(
    county_df: pd.DataFrame,
    flat_lam: np.ndarray,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Draw ONE realization of λ_c for all counties from the joint posterior.

    Parameters
    ----------
    county_df : DataFrame
        County metadata (must match posterior ordering).
    flat_lam : np.ndarray
        Flattened posterior samples (n_samples, n_counties).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    df_draw : DataFrame
        Columns: ['state', 'fips', 'lambda_draw'] plus any original county_df columns.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_counties = flat_lam.shape

    if len(county_df) != n_counties:
        raise ValueError(
            f"county_df has {len(county_df)} rows but posterior has {n_counties} counties. "
            "They must match in length and ordering."
        )

    idx = rng.integers(0, n_samples)
    lam_vec = flat_lam[idx, :]  # shape (n_counties,)

    df_draw = county_df.copy()
    df_draw["lambda_draw"] = lam_vec
    return df_draw


def aggregate_state_lambda(df_draw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate county-level λ_draw to state-level λ (simple county average).

    Parameters
    ----------
    df_draw : DataFrame
        Must contain columns ['state', 'lambda_draw'].

    Returns
    -------
    state_df : DataFrame
        Columns: ['state', 'n_counties', 'lambda_state_mean_draw'].

    Notes
    -----
    - 这里先用“按县均值”作为州 λ 的一个简单定义。
      如果以后需要按人口加权，只要在这里改成加权平均即可。
    """
    if "state" not in df_draw.columns:
        raise ValueError("df_draw must contain a 'state' column for aggregation.")

    grouped = df_draw.groupby("state", observed=True)["lambda_draw"]
    state_df = grouped.agg(
        lambda_state_mean_draw="mean",
        n_counties="count",
    ).reset_index()

    return state_df


def draw_state_event_counts(state_df: pd.DataFrame, random_state: int | None = None) -> pd.DataFrame:
    """
    For each state, draw the annual number of outage events from Poisson(λ_s).

    Parameters
    ----------
    state_df : DataFrame
        Must contain column 'lambda_state_mean_draw' (events/year).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    state_df_with_n : DataFrame
        Original columns plus 'n_events_draw' (int).
    """
    if "lambda_state_mean_draw" not in state_df.columns:
        raise ValueError("state_df must contain 'lambda_state_mean_draw' column.")

    rng = np.random.default_rng(random_state)
    lam = state_df["lambda_state_mean_draw"].values
    n_events = rng.poisson(lam)  # T = 1 year, so n ~ Poisson(λ_s)

    state_df = state_df.copy()
    state_df["n_events_draw"] = n_events
    return state_df


def prepare_outages_for_bootstrap(outages_df: pd.DataFrame) -> pd.DataFrame:
    """Add duration_hours and ACO (affected-customer-outage hours) columns."""
    df = outages_df.copy()
    if "fips" in df.columns:
        df["fips_str"] = df["fips"].astype(int).astype(str).str.zfill(5)
    if "duration_hours" not in df.columns:
        if "duration_min" in df.columns:
            df["duration_hours"] = df["duration_min"].astype(float) / 60.0
        else:
            df["duration_hours"] = 1.0
    cust_col = (
        "mean_customers"
        if "mean_customers" in df.columns
        else ("affected_customers" if "affected_customers" in df.columns else None)
    )
    if cust_col is None:
        raise ValueError("outages_df must contain mean_customers or affected_customers.")
    df["aco"] = df[cust_col].astype(float) * df["duration_hours"].astype(float)
    return df


def filter_top_aco_pool(events: pd.DataFrame, top_pct: float = 0.10) -> pd.DataFrame:
    """Keep the top `top_pct` fraction of events by ACO (at least one row)."""
    if events.empty:
        return events
    n_keep = max(1, int(np.ceil(len(events) * top_pct)))
    return events.nlargest(n_keep, "aco").copy()


def bootstrap_state_outages(
    state_df: pd.DataFrame,
    county_draw_df: pd.DataFrame,
    outages_df: pd.DataFrame,
    random_state: int | None = None,
    *,
    event_pool: str = "all",
    severe_top_pct: float = 0.10,
) -> pd.DataFrame:
    """
    For each state, bootstrap outage events n_s times using historical data.

    Here we ignore counties for simplicity and sample from that state's
    cleaned outage events (each row = one outage occurrence).

    Parameters
    ----------
    state_df : DataFrame
        Must contain ['state', 'n_events_draw'].
    county_draw_df : DataFrame
        County-level λ_draw realization with at least ['state', 'fips', 'lambda_draw'].
    outages_df : DataFrame
        Cleaned outage occurrence data with at least a 'state' column.
    random_state : int, optional
        Seed for reproducibility.
    event_pool : str
        'all' = full historical pool (normal scenario);
        'severe_top10' = only top `severe_top_pct` ACO events per county/state pool.
    severe_top_pct : float
        Top fraction by ACO when event_pool='severe_top10' (default 0.10).

    Returns
    -------
    boot_df : DataFrame
        Bootstrapped outage events across all states, with columns:
        - 'sim_state': state name
        - 'sim_county_fips': 5-digit county FIPS used for this event
        - 'sim_event_id': integer event id
    """
    if not {"state", "n_events_draw"}.issubset(state_df.columns):
        raise ValueError("state_df must contain 'state' and 'n_events_draw' columns.")
    if not {"state", "fips", "lambda_draw"}.issubset(county_draw_df.columns):
        raise ValueError(
            "county_draw_df must contain 'state', 'fips', and 'lambda_draw' columns."
        )
    if "state" not in outages_df.columns:
        raise ValueError("outages_df must contain a 'state' column.")
    event_pool = event_pool.lower().strip()
    if event_pool not in ("all", "severe_top10"):
        raise ValueError(f"event_pool must be 'all' or 'severe_top10', got {event_pool!r}")

    rng = np.random.default_rng(random_state)
    boot_list: list[pd.DataFrame] = []

    # Precompute county FIPS strings in both tables
    county_draw_df = county_draw_df.copy()
    county_draw_df["fips_str"] = county_draw_df["fips"].astype(int).astype(str).str.zfill(5)
    outages_df = prepare_outages_for_bootstrap(outages_df)

    for _, row in state_df.iterrows():
        state = row["state"]
        n_events = int(max(row["n_events_draw"], 0))
        if n_events == 0:
            continue

        # County-level λ_draw within this state
        state_counties = county_draw_df[county_draw_df["state"] == state]
        if state_counties.empty:
            continue

        lam_vec = state_counties["lambda_draw"].values
        lam_vec = np.clip(lam_vec, 1e-8, None)
        probs = lam_vec / lam_vec.sum()
        county_ids = state_counties["fips_str"].tolist()

        # Historical outages for this state
        state_events = outages_df[outages_df["state"] == state]
        if state_events.empty:
            continue
        if event_pool == "severe_top10":
            state_events = filter_top_aco_pool(state_events, severe_top_pct)

        # Group outages by county for faster lookup
        if event_pool == "severe_top10":
            groups = {
                k: filter_top_aco_pool(v, severe_top_pct)
                for k, v in state_events.groupby("fips_str", observed=True)
            }
        else:
            groups = dict(tuple(state_events.groupby("fips_str", observed=True)))

        sampled_rows: list[pd.DataFrame] = []
        for _ in range(n_events):
            # 1) sample county index according to λ_draw
            c_idx = rng.choice(len(county_ids), p=probs)
            c_fips = county_ids[c_idx]

            # 2) pick historical outage from that county; if none, fall back to state pool
            if c_fips in groups and len(groups[c_fips]) > 0:
                county_events = groups[c_fips]
                used_fallback = False
            else:
                county_events = state_events
                used_fallback = True
            if county_events.empty:
                continue
            j = rng.integers(0, len(county_events))
            row_s = county_events.iloc[[j]].copy()
            row_s["sim_state"] = state
            row_s["sim_county_fips"] = c_fips
            row_s["used_state_pool_fallback"] = used_fallback
            sampled_rows.append(row_s)

        if sampled_rows:
            boot_list.append(pd.concat(sampled_rows, axis=0, ignore_index=True))

    if not boot_list:
        return pd.DataFrame()

    boot_df = pd.concat(boot_list, axis=0, ignore_index=True)
    boot_df["sim_event_id"] = np.arange(len(boot_df))
    return boot_df


def draw_state_lambda_realization(
    county_df: pd.DataFrame,
    flat_lam: np.ndarray,
    state_name: str,
    random_state: int | None = None,
) -> Dict[str, float]:
    """
    Draw one λ realization for a specific state; returns dict[FIPS] -> λ_c.
    """
    df_draw = draw_lambda_realization(county_df, flat_lam, random_state=random_state)

    if "state" not in df_draw.columns:
        raise ValueError("county_estimates_advi.csv must contain a 'state' column.")

    df_state = df_draw[df_draw["state"] == state_name].copy()
    if df_state.empty:
        raise ValueError(f"No counties found for state: {state_name}")

    # Normalize FIPS as 5-digit strings
    df_state["fips_str"] = df_state["fips"].astype(int).astype(str).str.zfill(5)
    return dict(zip(df_state["fips_str"], df_state["lambda_draw"]))


def main() -> None:
    """
    Example usage:
    - Draw one realization for all counties
    - Save to CSV for downstream simulations
    """
    county_df, flat_lam = load_posterior()
    df_draw = draw_lambda_realization(county_df, flat_lam)
    state_df = aggregate_state_lambda(df_draw)
    state_df = draw_state_event_counts(state_df)

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    results_dir = project_root / "advi_results_2018_2023"
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load cleaned outage events for bootstrap
    outages_path = project_root / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
    if not outages_path.exists():
        raise FileNotFoundError(f"Cleaned outages CSV not found: {outages_path}")
    outages_df = pd.read_csv(outages_path)

    boot_df = bootstrap_state_outages(state_df, df_draw, outages_df)

    out_county = results_dir / "lambda_draw_sample1.csv"
    df_draw.to_csv(out_county, index=False)

    out_state = results_dir / "lambda_state_draw_sample1.csv"
    state_df.to_csv(out_state, index=False)

    out_boot = results_dir / "bootstrapped_outages_sample1.csv"
    boot_df.to_csv(out_boot, index=False)

    print(f"\n✓ Saved one county-level λ realization to: {out_county}")
    print(df_draw[["state", "fips", "lambda_draw"]].head())

    print(f"\n✓ Saved aggregated state-level λ with Poisson n_events to: {out_state}")
    print(state_df.head())

    print(f"\n✓ Saved bootstrapped outage events to: {out_boot}")
    print(boot_df[["sim_state"]].value_counts().head())


if __name__ == "__main__":
    main()

