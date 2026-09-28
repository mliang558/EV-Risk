"""Run one Batch-2 (unit × family × years) baseline job; reuse Batch-1 loss/cache."""

from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from attack_with_bootstrapped_outages import load_network
from counterfactual_batch1.crn_engine import (
    evaluate_events_on_graph,
    aggregate_sim_metrics,
)
from counterfactual_batch1.scenarios import Scenario
from counterfactual_batch1.urban_counties import urban_fips_from_mcc
from monte_carlo_outage_attack import (
    compute_radius_for_state_events,
    load_county_geometry_and_mcc,
    load_county_polygons,
    load_coverage_history,
    sample_point_in_polygon,
)
from sample_lambda_from_posterior import draw_lambda_realization, load_posterior
from select_epicenter_by_population import (
    choose_epicenter_population_mode,
    require_pop_units,
)
from compute_impact_radius import STATE_NAME_TO_ABBR

from .rates import build_r_table, ensure_outage_year, r_lookup
from .sampling import (
    aggregate_state_lambda_sum,
    bootstrap_year_outages,
    crn_seed,
    draw_state_event_counts_sum,
    scale_lambda_for_year,
)


def _git_commit(project_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _duration_hours(row: pd.Series) -> float:
    if "duration_hours" in row and pd.notna(row.get("duration_hours")):
        return float(row["duration_hours"])
    if "duration_min" in row and pd.notna(row.get("duration_min")):
        return float(row["duration_min"]) / 60.0
    return 1.0


def draw_crn_events_batch2(
    *,
    family: str,
    unit: str,
    sim_id: int,
    year_outage: int | None,
    target_states: list[str],
    outages_all: pd.DataFrame,
    r_table: pd.DataFrame,
    county_post_df: pd.DataFrame,
    flat_lam: Any,
    county_geom_mcc: pd.DataFrame,
    cov_df: pd.DataFrame,
    poly_dict: dict,
    pop_gdf: Any,
) -> pd.DataFrame:
    """
    One CRN event table for Batch-2.
    year_outage=None → family C pooled outages + unscaled λ (Batch-1-like).
    """
    seed = crn_seed(
        family=family,
        unit=unit,
        year_outage="pooled" if year_outage is None else int(year_outage),
        sim_id=sim_id,
    )
    df_draw = draw_lambda_realization(county_post_df, flat_lam, random_state=seed)

    if year_outage is None:
        # C: no r scaling; still use Σ_c for n ~ Poisson (Batch-2 C rate definition)
        df_scaled = df_draw.copy()
        df_scaled["r_st"] = 1.0
        outages_year = outages_all
    else:
        df_scaled = scale_lambda_for_year(df_draw, r_table, int(year_outage))
        outages_year = outages_all[outages_all["year"] == int(year_outage)].copy()

    state_df = aggregate_state_lambda_sum(df_scaled)
    state_df = draw_state_event_counts_sum(state_df, random_state=seed)

    boot = bootstrap_year_outages(
        state_df,
        df_scaled,
        outages_year=outages_year,
        outages_pooled=outages_all,
        random_state=seed,
    )
    boot_state = boot[boot["sim_state"].isin(target_states)].copy()
    if boot_state.empty:
        return pd.DataFrame()

    if "year" not in boot_state.columns:
        boot_state["year"] = pd.to_datetime(boot_state["start_time"]).dt.year

    chunks = []
    for ev_state in boot_state["sim_state"].unique():
        chunk = boot_state[boot_state["sim_state"] == ev_state]
        chunks.append(
            compute_radius_for_state_events(chunk, county_geom_mcc, cov_df, str(ev_state))
        )
    boot_state = pd.concat(chunks, ignore_index=True)

    # Attach r_st for the state's outage year (C → 1.0)
    if year_outage is None:
        boot_state["r_st"] = 1.0
    else:
        boot_state["r_st"] = [
            r_lookup(r_table, str(s), int(year_outage)) for s in boot_state["sim_state"]
        ]

    rng = np.random.default_rng(seed)
    epic_lats: list[float] = []
    epic_lons: list[float] = []
    for _, row in boot_state.iterrows():
        fips_str = str(row.get("fips_str") or row.get("sim_county_fips")).zfill(5)
        poly = poly_dict.get(fips_str)
        epic, _ = choose_epicenter_population_mode(
            county_fips=fips_str, poly=poly, rng=rng, pop_gdf=pop_gdf
        )
        if epic is None:
            if poly is not None:
                p = sample_point_in_polygon(poly, rng)
                epic_lats.append(float(p.y))
                epic_lons.append(float(p.x))
            else:
                epic_lats.append(np.nan)
                epic_lons.append(np.nan)
        else:
            epic_lats.append(float(epic[0]))
            epic_lons.append(float(epic[1]))

    boot_state["epicenter_lat"] = epic_lats
    boot_state["epicenter_lon"] = epic_lons
    boot_state = boot_state.dropna(subset=["epicenter_lat", "epicenter_lon"])
    if boot_state.empty:
        return pd.DataFrame()

    boot_state = boot_state.reset_index(drop=True)
    boot_state["sim_id"] = int(sim_id)
    boot_state["event_id"] = np.arange(len(boot_state), dtype=int)
    boot_state["seed"] = int(seed)
    boot_state["duration_hours"] = [
        _duration_hours(boot_state.iloc[i]) for i in range(len(boot_state))
    ]
    boot_state["radius_base_km"] = boot_state["impact_radius_km"].astype(float)
    boot_state["family"] = family
    boot_state["year_outage"] = "pooled" if year_outage is None else int(year_outage)
    return boot_state


def _metrics_row(
    m: dict[str, Any],
    *,
    family: str,
    year_network: int,
    year_outage: int | str,
    n_nodes: int,
    r_st: float,
    n_fb_county: int,
    n_fb_state_year: int,
    n_fb_state_pooled: int,
    git_commit: str,
) -> dict[str, Any]:
    row = dict(m)
    # aliases requested in Batch-2 spec
    row["L_cum"] = row.get("L_tilde")
    row["family"] = family
    row["year_network"] = int(year_network)
    row["year_outage"] = year_outage
    row["n_nodes"] = int(n_nodes)
    row["r_st"] = float(r_st)
    row["n_fb_county"] = int(n_fb_county)
    row["n_fb_state_year"] = int(n_fb_state_year)
    row["n_fb_state_pooled"] = int(n_fb_state_pooled)
    row["git_commit"] = git_commit
    return row


def run_unit_batch2(
    *,
    family: str,
    label: str,
    unit: str,
    member_states: list[str],
    pkl_path: Path,
    out_dir: Path,
    year_network: int,
    year_outage: int | None,
    n_sims: int,
    project_root: Path,
    force: bool = False,
    write_events: bool = False,
) -> str:
    """
    Baseline-only Batch-2 run for one unit.
    year_outage=None → family C (pooled outages).
    """
    family = family.upper()
    unit_out = out_dir
    unit_out.mkdir(parents=True, exist_ok=True)
    summary_path = unit_out / "summary.csv"
    if summary_path.is_file() and not force:
        return f"skip:{family}:{unit}:net{year_network}"

    G = load_network(pkl_path)
    n_nodes = int(G.number_of_nodes())
    try:
        E0 = float(nx.global_efficiency(G))
    except Exception:
        E0 = 0.0
    graph_id = f"net{year_network}:{unit}"

    county_post_df, flat_lam = load_posterior()
    outages_df = ensure_outage_year(
        pd.read_csv(
            project_root
            / "notebooks"
            / "PO_data_cleaning"
            / "cleaned_outages_2018_2023.csv"
        )
    )
    r_table = build_r_table(outages_df)
    county_geom_mcc = load_county_geometry_and_mcc(project_root)
    cov_df = load_coverage_history(project_root)
    county_polys = load_county_polygons(project_root)
    poly_dict = {row["fips_str"]: row["geometry"] for _, row in county_polys.iterrows()}
    urban_fips = urban_fips_from_mcc(county_geom_mcc, top_frac=0.25)
    pop_gdf = require_pop_units(project_root)
    git = _git_commit(project_root)

    target_states = list(member_states) if member_states else [label]
    for s in target_states:
        if STATE_NAME_TO_ABBR.get(s) is None:
            raise ValueError(f"No abbr mapping for {s}")

    # representative r_st for unit (mean over member states at year_outage)
    if year_outage is None:
        r_unit = 1.0
    else:
        rs = [r_lookup(r_table, s, int(year_outage)) for s in target_states]
        r_unit = float(np.mean(rs)) if rs else 1.0

    scenario = Scenario(
        key="baseline",
        family="baseline",
        description=f"batch2 {family} net={year_network} out={year_outage}",
    )
    removal_cache: OrderedDict = OrderedDict()
    metrics_rows: list[dict] = []
    yo_label: int | str = "pooled" if year_outage is None else int(year_outage)

    for sim_id in range(int(n_sims)):
        events = draw_crn_events_batch2(
            family=family,
            unit=unit,
            sim_id=sim_id,
            year_outage=year_outage,
            target_states=target_states,
            outages_all=outages_df,
            r_table=r_table,
            county_post_df=county_post_df,
            flat_lam=flat_lam,
            county_geom_mcc=county_geom_mcc,
            cov_df=cov_df,
            poly_dict=poly_dict,
            pop_gdf=pop_gdf,
        )
        if events.empty:
            continue
        # fallback counts for this sim
        fl = events["fallback_level"] if "fallback_level" in events.columns else None
        if fl is None:
            n_fb_c = n_fb_sy = n_fb_sp = 0
        else:
            n_fb_c = int((fl == "county").sum())
            n_fb_sy = int((fl == "state_year").sum())
            n_fb_sp = int((fl == "state_pooled").sum())

        ev = evaluate_events_on_graph(
            events=events,
            G=G,
            E0=E0,
            scenario=scenario,
            urban_fips=urban_fips,
            removal_cache=removal_cache,
            graph_id=graph_id,
        )
        m = aggregate_sim_metrics(ev)
        metrics_rows.append(
            _metrics_row(
                m,
                family=family,
                year_network=year_network,
                year_outage=yo_label,
                n_nodes=n_nodes,
                r_st=r_unit,
                n_fb_county=n_fb_c,
                n_fb_state_year=n_fb_sy,
                n_fb_state_pooled=n_fb_sp,
                git_commit=git,
            )
        )
        if write_events:
            ev.to_csv(
                unit_out / f"events_sim{sim_id}.csv.gz",
                index=False,
                compression="gzip",
            )

    if not metrics_rows:
        return f"empty:{family}:{unit}"

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(unit_out / "metrics_baseline.csv", index=False)
    summary = {
        "unit": unit,
        "label": label,
        "family": family,
        "year_network": int(year_network),
        "year_outage": yo_label,
        "n_sims": int(len(mdf)),
        "n_nodes": n_nodes,
        "r_st": r_unit,
        "L_cum_mean": float(mdf["L_cum"].mean()),
        "L_cum_std": float(mdf["L_cum"].std(ddof=1)) if len(mdf) > 1 else float("nan"),
        "total_rel_mean": float(mdf["total_rel_loss_x_duration"].mean()),
        "P_hit_mean": float(mdf["P_hit"].mean()) if "P_hit" in mdf else float("nan"),
        "n_fb_county": int(mdf["n_fb_county"].sum()),
        "n_fb_state_year": int(mdf["n_fb_state_year"].sum()),
        "n_fb_state_pooled": int(mdf["n_fb_state_pooled"].sum()),
        "git_commit": git,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    (unit_out / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "pkl_path": str(pkl_path),
                "member_states": target_states,
                "seed_rule": (
                    "C: sim_id (Batch1-compatible); "
                    "A/B: md5(unit|year_outage|sim)[:8]"
                ),
                "removal_cache_entries": len(removal_cache),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"ok:{family}:{unit}:net{year_network}:out{yo_label}"
