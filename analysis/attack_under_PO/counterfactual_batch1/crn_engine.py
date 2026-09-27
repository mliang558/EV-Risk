#!/usr/bin/env python3
"""
Common-random-numbers (CRN) multi-scenario Monte Carlo engine.

For each sim_id:
  1. Draw λ, n_events, bootstrap outages, epicenters ONCE.
  2. Evaluate every scenario on that same event list
     (scale R_c / T_i, or swap graph for CF-D / K).
  3. Cache E(G) per graph variant.
  4. Write event-level + sim-level outputs (new result roots only).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from attack_with_bootstrapped_outages import (
    _affected_nodes_in_radius,
    compute_event_loss,
)
from compute_impact_radius import (
    load_county_geometry_and_mcc,
    load_coverage_history,
    STATE_NAME_TO_ABBR,
)
from monte_carlo_outage_attack import (
    compute_radius_for_state_events,
    load_county_polygons,
    load_network,
    sample_point_in_polygon,
)
from sample_lambda_from_posterior import (
    aggregate_state_lambda,
    bootstrap_state_outages,
    draw_lambda_realization,
    draw_state_event_counts,
    load_county_estimates,
    load_posterior,
)
from select_epicenter_by_stations import choose_station_epicenter_in_county
from select_epicenter_by_population import (
    choose_epicenter_population_mode,
    load_pop_units_if_available,
)

from counterfactual_batch1.densify_network import load_or_build_densified
from counterfactual_batch1.scenarios import Scenario
from counterfactual_batch1.urban_counties import urban_fips_from_mcc


def _duration_hours(row: pd.Series) -> float:
    if "duration_hours" in row and pd.notna(row.get("duration_hours")):
        return float(row["duration_hours"])
    if "duration_min" in row and pd.notna(row.get("duration_min")):
        return float(row["duration_min"]) / 60.0
    return 1.0


def _filter_outages_by_year(outages_df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """If year is set, keep only that calendar year's historical events."""
    if year is None:
        return outages_df
    df = outages_df.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["start_time"]).dt.year
    return df[df["year"].astype(int) == int(year)].copy()


def draw_crn_events(
    *,
    sim_id: int,
    target_states: list[str],
    outages_df: pd.DataFrame,
    county_post_df: pd.DataFrame,
    flat_lam: Any,
    county_geom_mcc: pd.DataFrame,
    cov_df: pd.DataFrame,
    poly_dict: dict,
    G_for_epicenter: nx.Graph,
    bootstrap_pool: str = "all",
    outage_year: int | None = None,
    epicenter_mode: str = "population",
    pop_gdf: Any = None,
    kde_sigma_km: float = 3.0,
) -> pd.DataFrame:
    """
    One CRN draw: λ → n_events → bootstrap → radius → epicenter.

    Epicenter (lat, lon) frozen for all scenarios (CRN). Default population mode
    uses tract pop units or station-KDE on the baseline graph; CF densify nodes
    never affect epicenter placement.
    """
    df_draw = draw_lambda_realization(county_post_df, flat_lam, random_state=sim_id)
    state_df = aggregate_state_lambda(df_draw)
    state_df = draw_state_event_counts(state_df, random_state=sim_id)

    pool = _filter_outages_by_year(outages_df, outage_year)
    boot_all = bootstrap_state_outages(
        state_df,
        df_draw,
        pool,
        random_state=sim_id,
        event_pool=bootstrap_pool,
    )
    boot_state = boot_all[boot_all["sim_state"].isin(target_states)].copy()
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
    boot_state = pd.concat(chunks, axis=0, ignore_index=True)

    rng = np.random.default_rng(sim_id)
    epic_lats: list[float] = []
    epic_lons: list[float] = []
    mode = (epicenter_mode or "population").lower().strip()
    for _, row in boot_state.iterrows():
        fips_str = str(row["fips_str"]).zfill(5)
        poly = poly_dict.get(fips_str)

        if mode == "station":
            station_epic = choose_station_epicenter_in_county(
                G_for_epicenter,
                county_fips=fips_str,
                county_attr="county_fips",
                weight_attr=None,
                rng=rng,
                exclude_cf_added=True,
            )
            if station_epic is not None:
                epic_lats.append(float(station_epic[0]))
                epic_lons.append(float(station_epic[1]))
                continue
            if poly is None:
                epic_lats.append(np.nan)
                epic_lons.append(np.nan)
                continue
            p = sample_point_in_polygon(poly, rng)
            epic_lats.append(float(p.y))
            epic_lons.append(float(p.x))
            continue

        epic, _meth = choose_epicenter_population_mode(
            county_fips=fips_str,
            poly=poly,
            rng=rng,
            G=G_for_epicenter,
            pop_gdf=pop_gdf,
            sigma_km=kde_sigma_km,
        )
        if epic is None:
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
    boot_state["seed"] = int(sim_id)
    boot_state["duration_hours"] = [
        _duration_hours(boot_state.iloc[i]) for i in range(len(boot_state))
    ]
    boot_state["radius_base_km"] = boot_state["impact_radius_km"].astype(float)
    return boot_state


def _effective_radius(
    row: pd.Series,
    scenario: Scenario,
    urban_fips: set[str],
) -> float:
    r = float(row["radius_base_km"])
    if scenario.urban_radius_scale is not None:
        fips = str(row.get("fips_str", "")).zfill(5)
        if fips in urban_fips:
            return r * float(scenario.urban_radius_scale)
        return r
    return r * float(scenario.radius_scale)


def evaluate_events_on_graph(
    *,
    events: pd.DataFrame,
    G: nx.Graph,
    E0: float,
    scenario: Scenario,
    urban_fips: set[str],
) -> pd.DataFrame:
    """Apply one scenario to a shared CRN event table; return event-level rows."""
    n_nodes = G.number_of_nodes()
    records: list[dict] = []

    for _, row in events.iterrows():
        radius_km = _effective_radius(row, scenario, urban_fips)
        duration_h = float(row["duration_hours"]) * float(scenario.duration_scale)
        epicenter = (float(row["epicenter_lat"]), float(row["epicenter_lon"]))

        eff_before, eff_after, eff_loss, pct_loss, lcc_frac = compute_event_loss(
            G, epicenter, radius_km, eff_before=E0
        )
        if pct_loss is not None and pct_loss < 0:
            eff_loss = 0.0
            pct_loss = 0.0

        n_disrupted = len(_affected_nodes_in_radius(G, epicenter, radius_km))
        lcc_deficit = max(0.0, 1.0 - float(lcc_frac))
        rel_loss_x_duration = float(pct_loss) * duration_h
        rel_lcc_loss_x_duration = lcc_deficit * duration_h
        nd = float(row.get("Nd", row.get("affected_customers", 0.0)) or 0.0)

        records.append(
            {
                "sim_id": int(row["sim_id"]),
                "event_id": int(row["event_id"]),
                "seed": int(row["seed"]),
                "scenario": scenario.key,
                "family": scenario.family,
                "year_hist": int(row["year"]),
                "fips_str": str(row["fips_str"]).zfill(5),
                "epicenter_lat": epicenter[0],
                "epicenter_lon": epicenter[1],
                "radius_base_km": float(row["radius_base_km"]),
                "radius_km": float(radius_km),
                "duration_hours": float(duration_h),
                "duration_base_hours": float(row["duration_hours"]),
                "pct_eff_loss": float(pct_loss),
                "eff_loss": float(eff_loss),
                "rel_loss_x_duration": float(rel_loss_x_duration),
                "rel_lcc_loss_x_duration": float(rel_lcc_loss_x_duration),
                "lcc_frac": float(lcc_frac),
                "n_disrupted_nodes": int(n_disrupted),
                "customer_hours": nd * float(row["duration_hours"]),
                "used_state_pool_fallback": bool(
                    row.get("used_state_pool_fallback", False)
                ),
                "n_nodes": int(n_nodes),
                "baseline_efficiency": float(E0),
            }
        )
    return pd.DataFrame(records)


def aggregate_sim_metrics(event_df: pd.DataFrame) -> dict[str, Any]:
    if event_df.empty:
        return {}
    n_nodes = int(event_df["n_nodes"].iloc[0])
    total_rel = float(event_df["rel_loss_x_duration"].sum())
    L_tilde = total_rel / n_nodes if n_nodes > 0 else float("nan")
    total_lcc = float(event_df["rel_lcc_loss_x_duration"].sum())
    L_lcc = total_lcc / n_nodes if n_nodes > 0 else float("nan")
    n_ev = int(len(event_df))
    n_zero = int((event_df["pct_eff_loss"] <= 0).sum())
    n_hit = max(0, n_ev - n_zero)
    p_hit = float(n_hit / n_ev) if n_ev > 0 else float("nan")
    L_event = total_rel / n_ev if n_ev > 0 else float("nan")
    E_loss_hit = total_rel / n_hit if n_hit > 0 else float("nan")
    return {
        "sim_id": int(event_df["sim_id"].iloc[0]),
        "scenario": str(event_df["scenario"].iloc[0]),
        "family": str(event_df["family"].iloc[0]),
        "n_events": n_ev,
        "n_nodes": n_nodes,
        "baseline_efficiency": float(event_df["baseline_efficiency"].iloc[0]),
        "L_tilde": L_tilde,
        "L_lcc_tilde": L_lcc,
        "total_rel_loss_x_duration": total_rel,
        "mean_pct_eff_loss": float(event_df["pct_eff_loss"].mean()),
        "mean_lcc_frac": float(event_df["lcc_frac"].mean()),
        "n_disrupted_nodes_mean": float(event_df["n_disrupted_nodes"].mean()),
        "n_zero_loss_events": n_zero,
        "n_hit_events": n_hit,
        "P_hit": p_hit,
        "L_event": L_event,
        "E_loss_given_hit": E_loss_hit,
        "n_fallback_events": int(event_df["used_state_pool_fallback"].sum()),
        "max_disruption_radius_km": float(event_df["radius_km"].max()),
    }


def resolve_graph_for_scenario(
    scenario: Scenario,
    *,
    G_base: nx.Graph,
    unit: str,
    densify_cache: Path,
    k5km_pkl: Path | None,
    densify_seed: int = 42,
    member_state_abbrs: list[str] | None = None,
) -> tuple[nx.Graph | None, str]:
    """
    Return (graph, note). None graph → skip scenario (e.g. missing 5 km pickle).

    Densified graphs are built from the existing baseline 10 km pickle (G_base);
    they are never used for epicenter sampling.
    """
    if scenario.network_variant == "base_10km":
        return G_base, "base_10km"
    if scenario.network_variant == "k5km":
        if k5km_pkl is None or not k5km_pkl.is_file():
            return None, "missing_k5km"
        return load_network(k5km_pkl), "k5km"
    if scenario.densify_pct is not None and scenario.densify_mode:
        H, _meta = load_or_build_densified(
            G_base,
            unit,
            scenario.densify_pct,
            scenario.densify_mode,
            densify_cache,
            seed=int(getattr(scenario, "densify_seed", densify_seed)),
            replicate=getattr(scenario, "densify_replicate", None),
            member_state_abbrs=member_state_abbrs,
        )
        return H, scenario.network_variant
    return G_base, scenario.network_variant


def run_unit_batch1(
    *,
    label: str,
    unit: str,
    member_states: list[str],
    pkl_path: Path,
    out_dir: Path,
    scenarios: list[Scenario],
    n_sims: int = 100,
    project_root: Path,
    densify_cache: Path,
    k5km_pkl: Path | None = None,
    bootstrap_pool: str = "all",
    outage_year: int | None = None,
    force: bool = False,
    write_events: bool = True,
    epicenter_mode: str = "population",
    kde_sigma_km: float = 3.0,
) -> str:
    """
    Run all Batch-1 scenarios for one analysis unit with shared CRN draws.

    Outputs under out_dir / <unit>/ :
      crn_events.csv.gz          — shared event draws (once)
      events_<scenario>.csv.gz   — per-scenario event losses
      metrics_<scenario>.csv     — per-sim L_tilde etc.
      summary.csv                — pooled means + ΔL vs baseline
      manifest.json
    """
    unit_out = out_dir / unit
    unit_out.mkdir(parents=True, exist_ok=True)
    summary_path = unit_out / "summary.csv"
    if summary_path.is_file() and not force:
        return f"skip:{unit}"

    G_base = load_network(pkl_path)
    target_states = list(member_states) if member_states else [label]
    member_abbrs: list[str] = []
    for s in target_states:
        if STATE_NAME_TO_ABBR.get(s) is None:
            raise ValueError(f"No abbr mapping for {s}")
        member_abbrs.append(STATE_NAME_TO_ABBR[s])

    county_post_df, flat_lam = load_posterior()
    outages_df = pd.read_csv(
        project_root / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
    )
    county_geom_mcc = load_county_geometry_and_mcc(project_root)
    cov_df = load_coverage_history(project_root)
    county_polys = load_county_polygons(project_root)
    poly_dict = {row["fips_str"]: row["geometry"] for _, row in county_polys.iterrows()}
    urban_fips = urban_fips_from_mcc(county_geom_mcc, top_frac=0.25)
    pop_gdf = (
        load_pop_units_if_available(project_root)
        if (epicenter_mode or "population").lower() == "population"
        else None
    )

    # Preload / build graphs + cache E(G)
    # Epicenters always drawn once per sim (population/KDE on G_base); densified
    # graphs only for loss evaluation.
    graph_cache: dict[str, tuple[nx.Graph, float]] = {}
    skip_keys: set[str] = set()
    for sc in scenarios:
        G_sc, note = resolve_graph_for_scenario(
            sc,
            G_base=G_base,
            unit=unit,
            densify_cache=densify_cache,
            k5km_pkl=k5km_pkl,
            member_state_abbrs=member_abbrs,
        )
        if G_sc is None:
            skip_keys.add(sc.key)
            print(f"[skip-scenario] {unit} {sc.key}: {note}", flush=True)
            continue
        variant = note
        if variant not in graph_cache:
            try:
                E0 = float(nx.global_efficiency(G_sc))
            except Exception:
                E0 = 0.0
            graph_cache[variant] = (G_sc, E0)

    active = [s for s in scenarios if s.key not in skip_keys]
    if not active:
        return f"empty-scenarios:{unit}"

    crn_rows: list[pd.DataFrame] = []
    metrics_by_sc: dict[str, list[dict]] = {s.key: [] for s in active}
    events_by_sc: dict[str, list[pd.DataFrame]] = {s.key: [] for s in active}

    t0 = time.time()
    for sim_id in range(n_sims):
        # CRITICAL CRN lock: epicenter coords frozen; population mode on baseline G
        events = draw_crn_events(
            sim_id=sim_id,
            target_states=target_states,
            outages_df=outages_df,
            county_post_df=county_post_df,
            flat_lam=flat_lam,
            county_geom_mcc=county_geom_mcc,
            cov_df=cov_df,
            poly_dict=poly_dict,
            G_for_epicenter=G_base,
            bootstrap_pool=bootstrap_pool,
            outage_year=outage_year,
            epicenter_mode=epicenter_mode,
            pop_gdf=pop_gdf,
            kde_sigma_km=kde_sigma_km,
        )
        if events.empty:
            continue
        crn_rows.append(
            events[
                [
                    "sim_id",
                    "event_id",
                    "seed",
                    "year",
                    "fips_str",
                    "epicenter_lat",
                    "epicenter_lon",
                    "radius_base_km",
                    "duration_hours",
                    "used_state_pool_fallback",
                ]
            ].rename(columns={"year": "year_hist"})
        )

        for sc in active:
            G_sc, note = resolve_graph_for_scenario(
                sc,
                G_base=G_base,
                unit=unit,
                densify_cache=densify_cache,
                k5km_pkl=k5km_pkl,
                member_state_abbrs=member_abbrs,
            )
            assert G_sc is not None
            G_use, E0 = graph_cache[note]
            ev = evaluate_events_on_graph(
                events=events,
                G=G_use,
                E0=E0,
                scenario=sc,
                urban_fips=urban_fips,
            )
            if write_events:
                events_by_sc[sc.key].append(ev)
            metrics_by_sc[sc.key].append(aggregate_sim_metrics(ev))

        if (sim_id + 1) % 5 == 0 or (sim_id + 1) == n_sims:
            elapsed = time.time() - t0
            print(
                f"[CF-B1] {label} {sim_id+1}/{n_sims} "
                f"({100*(sim_id+1)/n_sims:.0f}%) elapsed {elapsed/60:.1f} min",
                flush=True,
            )

    if not crn_rows:
        return f"empty:{unit}"

    pd.concat(crn_rows, ignore_index=True).to_csv(
        unit_out / "crn_events.csv.gz", index=False, compression="gzip"
    )

    summary_rows: list[dict] = []
    baseline_mean = None
    for sc in active:
        mdf = pd.DataFrame(metrics_by_sc[sc.key])
        mdf.to_csv(unit_out / f"metrics_{sc.key}.csv", index=False)
        if write_events and events_by_sc[sc.key]:
            pd.concat(events_by_sc[sc.key], ignore_index=True).to_csv(
                unit_out / f"events_{sc.key}.csv.gz", index=False, compression="gzip"
            )
        mean_L = float(mdf["L_tilde"].mean()) if len(mdf) else float("nan")
        if sc.key == "baseline":
            baseline_mean = mean_L
        summary_rows.append(
            {
                "unit": unit,
                "label": label,
                "scenario": sc.key,
                "family": sc.family,
                "dose_pct": float(sc.dose_pct) if sc.dose_pct == sc.dose_pct else float("nan"),
                "n_sims": int(len(mdf)),
                "L_tilde_mean": mean_L,
                "L_tilde_std": float(mdf["L_tilde"].std(ddof=1)) if len(mdf) > 1 else float("nan"),
                "L_lcc_tilde_mean": float(mdf["L_lcc_tilde"].mean()) if len(mdf) else float("nan"),
                "n_nodes_mean": float(mdf["n_nodes"].mean()) if len(mdf) else float("nan"),
                "description": sc.description,
            }
        )

    for row in summary_rows:
        if baseline_mean is not None and np.isfinite(baseline_mean) and baseline_mean != 0:
            row["delta_L_tilde"] = float(row["L_tilde_mean"] - baseline_mean)
            row["delta_L_tilde_pct"] = float(
                100.0 * (row["L_tilde_mean"] - baseline_mean) / abs(baseline_mean)
            )
        else:
            row["delta_L_tilde"] = float("nan")
            row["delta_L_tilde_pct"] = float("nan")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    manifest = {
        "unit": unit,
        "label": label,
        "member_states": target_states,
        "n_sims": n_sims,
        "outage_year": outage_year,
        "bootstrap_pool": bootstrap_pool,
        "scenarios": [s.key for s in active],
        "skipped_scenarios": sorted(skip_keys),
        "pkl_path": str(pkl_path),
        "densify_cache": str(densify_cache),
    }
    (unit_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return f"ok:{unit}"
