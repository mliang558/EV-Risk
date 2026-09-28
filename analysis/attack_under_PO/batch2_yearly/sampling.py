"""Batch-2 CRN sampling: year pools, λ scaling, severity fallback ladder."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
import pandas as pd

from sample_lambda_from_posterior import (
    draw_lambda_realization,
    prepare_outages_for_bootstrap,
)

FallbackLevel = Literal["county", "state_year", "state_pooled"]


def crn_seed(*, family: str, unit: str, year_outage: int | str, sim_id: int) -> int:
    """
    A/B: hash(unit, t_outage, sim) — network year independent (A≡B events).
    C:   sim_id only — matches Batch-1 baseline CRN (seed = sim_id).
    """
    if family.upper() == "C":
        return int(sim_id)
    key = f"{unit}|{year_outage}|{int(sim_id)}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def scale_lambda_for_year(
    df_draw: pd.DataFrame,
    r_table: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """λ_c,t = λ_c × r[s,t]."""
    out = df_draw.copy()
    rmap = {
        (row["state"], int(row["year"])): float(row["r"])
        for _, row in r_table.iterrows()
    }
    scales = []
    for _, row in out.iterrows():
        scales.append(rmap.get((row["state"], int(year)), 1.0))
    out["r_st"] = scales
    out["lambda_draw"] = out["lambda_draw"].astype(float) * out["r_st"]
    return out


def aggregate_state_lambda_sum(df_draw: pd.DataFrame) -> pd.DataFrame:
    """λ_s = Σ_c λ_c (Batch-2 spec; differs from Batch-1 county-mean helper)."""
    g = df_draw.groupby("state", observed=True)["lambda_draw"]
    return g.agg(lambda_state_sum="sum", n_counties="count").reset_index()


def draw_state_event_counts_sum(
    state_df: pd.DataFrame, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    lam = state_df["lambda_state_sum"].to_numpy(dtype=float)
    out = state_df.copy()
    out["n_events_draw"] = rng.poisson(np.clip(lam, 0, None))
    return out


def _pool_maps(
    outages_year: pd.DataFrame,
    outages_pooled: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """county_year, state_year, state_pooled keyed by fips_str / state."""
    county_year = {
        k: v for k, v in outages_year.groupby("fips_str", observed=True) if len(v)
    }
    state_year = {
        k: v for k, v in outages_year.groupby("state", observed=True) if len(v)
    }
    state_pooled = {
        k: v for k, v in outages_pooled.groupby("state", observed=True) if len(v)
    }
    return county_year, state_year, state_pooled


def _pick_event(
    rng: np.random.Generator,
    *,
    c_fips: str,
    state: str,
    county_year: dict[str, pd.DataFrame],
    state_year: dict[str, pd.DataFrame],
    state_pooled: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame | None, FallbackLevel]:
    if c_fips in county_year and len(county_year[c_fips]) > 0:
        pool = county_year[c_fips]
        level: FallbackLevel = "county"
    elif state in state_year and len(state_year[state]) > 0:
        pool = state_year[state]
        level = "state_year"
    elif state in state_pooled and len(state_pooled[state]) > 0:
        pool = state_pooled[state]
        level = "state_pooled"
    else:
        return None, "state_pooled"
    j = int(rng.integers(0, len(pool)))
    return pool.iloc[[j]].copy(), level


def bootstrap_year_outages(
    state_df: pd.DataFrame,
    county_draw_df: pd.DataFrame,
    outages_year: pd.DataFrame,
    outages_pooled: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample n_s events per state with county ∝ λ_c.

    Severity fallback:
      1) county_year_pool[c,t]
      2) state_year_pool[s,t]
      3) 6-year state pool (flag)
    Event rows that share a calendar day remain separate pool entries
    (independent draws) — same as Batch-1 row-level bootstrap.
    """
    if state_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(int(random_state))
    county_draw_df = county_draw_df.copy()
    county_draw_df["fips_str"] = (
        county_draw_df["fips"].astype(int).astype(str).str.zfill(5)
    )
    outages_year = prepare_outages_for_bootstrap(outages_year)
    outages_pooled = prepare_outages_for_bootstrap(outages_pooled)
    county_year, state_year, state_pooled = _pool_maps(outages_year, outages_pooled)

    boot_list: list[pd.DataFrame] = []
    for _, row in state_df.iterrows():
        state = row["state"]
        n_events = int(max(row["n_events_draw"], 0))
        if n_events == 0:
            continue
        state_counties = county_draw_df[county_draw_df["state"] == state]
        if state_counties.empty:
            continue
        lam_vec = np.clip(state_counties["lambda_draw"].to_numpy(dtype=float), 1e-8, None)
        probs = lam_vec / lam_vec.sum()
        county_ids = state_counties["fips_str"].tolist()

        sampled: list[pd.DataFrame] = []
        for _ in range(n_events):
            c_idx = int(rng.choice(len(county_ids), p=probs))
            c_fips = county_ids[c_idx]
            picked, level = _pick_event(
                rng,
                c_fips=c_fips,
                state=state,
                county_year=county_year,
                state_year=state_year,
                state_pooled=state_pooled,
            )
            if picked is None or picked.empty:
                continue
            picked = picked.copy()
            picked["sim_state"] = state
            picked["sim_county_fips"] = c_fips
            picked["fallback_level"] = level
            picked["used_state_pool_fallback"] = level != "county"
            sampled.append(picked)
        if sampled:
            boot_list.append(pd.concat(sampled, ignore_index=True))

    if not boot_list:
        return pd.DataFrame()
    boot = pd.concat(boot_list, ignore_index=True)
    boot["sim_event_id"] = np.arange(len(boot))
    return boot
