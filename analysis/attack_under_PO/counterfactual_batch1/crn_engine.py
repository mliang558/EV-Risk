#!/usr/bin/env python3
"""Common-random-number engine for the two intervention experiments.

Each replication draws outage rates, events, and epicenters once. The baseline,
corridor networks, and severity doses are evaluated on that shared event list.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "analysis", ROOT / "analysis" / "attack_under_PO"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attack_with_bootstrapped_outages import (
    compute_event_loss,
)
from compute_impact_radius import (
    STATE_NAME_TO_ABBR,
    load_county_geometry_and_mcc,
    load_coverage_history,
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
    load_posterior,
)
from select_epicenter_by_stations import choose_station_epicenter_in_county
from select_epicenter_by_population import (
    choose_epicenter_population_mode,
    require_pop_units,
)

from counterfactual_batch1.densify_network import load_or_build_densified
from counterfactual_batch1.scenarios import Scenario


def _duration_hours(row: pd.Series) -> float:
    if "duration_hours" in row and pd.notna(row.get("duration_hours")):
        duration = float(row["duration_hours"])
    elif "duration_min" in row and pd.notna(row.get("duration_min")):
        duration = float(row["duration_min"]) / 60.0
    else:
        duration = 1.0
    if not np.isfinite(duration) or duration < 0:
        raise ValueError(f"Invalid event duration: {duration}")
    return duration


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
) -> pd.DataFrame:
    """
    One CRN draw: λ → n_events → bootstrap → radius → epicenter.

    Epicenter (lat, lon) frozen for all scenarios (CRN). Default population mode
    uses census-tract population only (network-independent). CF densify nodes
    never affect epicenter placement.
    """
    # The posterior draw is shared across analytical units for a joint national
    # realization. Count, event, and epicenter streams are unit-specific, while
    # remaining shared by all intervention scenarios within that unit.
    unit_key = "|".join(sorted(map(str, target_states))).encode("utf-8")
    unit_entropy = int.from_bytes(hashlib.sha256(unit_key).digest()[:8], "little")
    lambda_seed = int(
        np.random.SeedSequence(sim_id)
        .spawn(1)[0]
        .generate_state(1, dtype=np.uint32)[0]
    )
    stage_seqs = np.random.SeedSequence([sim_id, unit_entropy]).spawn(3)
    count_seed, bootstrap_seed, epicenter_seed = [
        int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in stage_seqs
    ]

    df_draw_all = draw_lambda_realization(
        county_post_df, flat_lam, random_state=lambda_seed
    )
    # Each worker evaluates one analytical unit. Restrict the draw before
    # constructing events; generating all nationwide events in every worker is
    # distributionally redundant and dominates runtime once state rates are
    # correctly defined as county-rate sums.
    df_draw = df_draw_all[df_draw_all["state"].isin(target_states)].copy()
    if df_draw.empty:
        raise ValueError(f"Posterior contains no counties for {target_states}")
    state_df = aggregate_state_lambda(df_draw)
    state_df = draw_state_event_counts(state_df, random_state=count_seed)
    expected_events = int(state_df["n_events_draw"].sum())
    if expected_events == 0:
        return pd.DataFrame()

    pool = _filter_outages_by_year(outages_df, outage_year)
    pool = pool[pool["state"].isin(target_states)].copy()
    boot_all = bootstrap_state_outages(
        state_df,
        df_draw,
        pool,
        random_state=bootstrap_seed,
        event_pool=bootstrap_pool,
    )
    if len(boot_all) != expected_events:
        raise RuntimeError(
            f"Expected {expected_events} bootstrapped events for {target_states}, "
            f"but constructed {len(boot_all)}"
        )
    boot_state = boot_all[boot_all["sim_state"].isin(target_states)].copy()

    if "year" not in boot_state.columns:
        boot_state["year"] = pd.to_datetime(boot_state["start_time"]).dt.year

    chunks = []
    for ev_state in boot_state["sim_state"].unique():
        chunk = boot_state[boot_state["sim_state"] == ev_state]
        state_abbr = STATE_NAME_TO_ABBR.get(str(ev_state))
        available_years = set(
            cov_df.loc[cov_df["state_abbr"] == state_abbr, "year_num"]
            .dropna()
            .astype(int)
        )
        missing_years = sorted(set(chunk["year"].astype(int)) - available_years)
        if state_abbr is None or missing_years:
            raise ValueError(
                f"Missing coverage history for {ev_state}: years={missing_years}"
            )
        chunks.append(
            compute_radius_for_state_events(chunk, county_geom_mcc, cov_df, str(ev_state))
        )
    boot_state = pd.concat(chunks, axis=0, ignore_index=True)
    radius_fields = ["Nd", "MCC", "rho", "theta_s", "impact_radius_km"]
    invalid_radius = (
        ~np.isfinite(boot_state[radius_fields].astype(float)).all(axis=1)
        | (boot_state["Nd"].astype(float) < 0)
        | (boot_state["MCC"].astype(float) <= 0)
        | (boot_state["rho"].astype(float) <= 0)
        | (boot_state["impact_radius_km"].astype(float) < 0)
    )
    if invalid_radius.any():
        bad = sorted(boot_state.loc[invalid_radius, "fips_str"].astype(str).unique())
        raise ValueError(f"Invalid radius inputs for county FIPS: {bad[:10]}")

    rng = np.random.default_rng(epicenter_seed)
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
            pop_gdf=pop_gdf,
        )
        if epic is None:
            epic_lats.append(np.nan)
            epic_lons.append(np.nan)
        else:
            epic_lats.append(float(epic[0]))
            epic_lons.append(float(epic[1]))

    boot_state["epicenter_lat"] = epic_lats
    boot_state["epicenter_lon"] = epic_lons
    missing_epicenter = boot_state[["epicenter_lat", "epicenter_lon"]].isna().any(axis=1)
    if missing_epicenter.any():
        bad = sorted(boot_state.loc[missing_epicenter, "fips_str"].astype(str).unique())
        raise ValueError(f"Could not place epicenters for county FIPS: {bad[:10]}")

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
) -> float:
    r = float(row["radius_base_km"])
    if scenario.affected_customer_scale is not None:
        required = ("Nd", "theta_s", "MCC", "rho")
        missing = [name for name in required if name not in row.index or pd.isna(row[name])]
        if missing:
            raise ValueError(
                "Cannot recompute severity-reduction radius; missing event fields: "
                + ", ".join(missing)
            )
        nd = max(0.0, float(row["Nd"])) * float(scenario.affected_customer_scale)
        theta = float(np.clip(float(row["theta_s"]), 1e-3, 1.0))
        mcc = max(0.0, float(row["MCC"]))
        rho = max(1e-6, float(row["rho"]))
        nd_effective = min(nd / theta, mcc)
        return float(np.sqrt(nd_effective / (rho * np.pi)))
    return r


def evaluate_events_on_graph(
    *,
    events: pd.DataFrame,
    G: nx.Graph,
    E0: float,
    scenario: Scenario,
    removal_cache: dict | None = None,
    graph_id: str | None = None,
) -> pd.DataFrame:
    """Apply one scenario to a shared CRN event table; return event-level rows."""
    import os
    from collections import OrderedDict

    n_nodes = G.number_of_nodes()
    records: list[dict] = []
    # Keys are (graph_id, frozenset(S)) — CF-D dose graphs never collide.
    # Disable: CF_B1_REMOVAL_CACHE=0. LRU size: CF_B1_REMOVAL_CACHE_MAX (default 50000).
    use_cache = os.environ.get("CF_B1_REMOVAL_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if not use_cache:
        removal_cache = None
    elif removal_cache is None:
        removal_cache = OrderedDict()
    cache_max = int(os.environ.get("CF_B1_REMOVAL_CACHE_MAX", "50000"))
    gid = graph_id or scenario.network_variant or scenario.key

    for _, row in events.iterrows():
        radius_km = _effective_radius(row, scenario)
        duration_h = float(row["duration_hours"])
        epicenter = (float(row["epicenter_lat"]), float(row["epicenter_lon"]))

        (
            eff_before,
            eff_after,
            eff_loss,
            pct_loss,
            lcc_frac,
            affected,
        ) = compute_event_loss(
            G,
            epicenter,
            radius_km,
            eff_before=E0,
            removal_cache=removal_cache,
            graph_id=gid,
            removal_cache_max=cache_max,
        )
        if pct_loss is not None and pct_loss < 0:
            eff_loss = 0.0
            pct_loss = 0.0

        n_disrupted = len(affected)
        hit = int(1 if n_disrupted > 0 else 0)
        if hit == 0:
            eff_loss = 0.0
            pct_loss = 0.0
        lcc_deficit = max(0.0, 1.0 - float(lcc_frac))
        rel_loss_x_duration = float(pct_loss) * duration_h
        rel_lcc_loss_x_duration = lcc_deficit * duration_h
        nd = float(row.get("Nd", row.get("affected_customers", 0.0)) or 0.0)
        nd_scale = (
            float(scenario.affected_customer_scale)
            if scenario.affected_customer_scale is not None
            else 1.0
        )
        nd_scenario = nd * nd_scale

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
                "hit": hit,
                "n_disrupted_nodes": int(n_disrupted),
                "radius_base_km": float(row["radius_base_km"]),
                "radius_km": float(radius_km),
                "duration_hours": float(duration_h),
                "duration_base_hours": float(row["duration_hours"]),
                "pct_eff_loss": float(pct_loss),
                "eff_loss": float(eff_loss),
                "rel_loss_x_duration": float(rel_loss_x_duration),
                "rel_lcc_loss_x_duration": float(rel_lcc_loss_x_duration),
                "lcc_frac": float(lcc_frac),
                "affected_customers_base": nd,
                "affected_customers_scenario": nd_scenario,
                "customer_hours": nd_scenario * float(row["duration_hours"]),
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
    n_hit = int(event_df["hit"].sum()) if "hit" in event_df.columns else 0
    n_zero = n_ev - n_hit
    p_hit = float(n_hit / n_ev) if n_ev > 0 else float("nan")
    L_event = total_rel / n_ev if n_ev > 0 else float("nan")
    if n_hit > 0 and "hit" in event_df.columns:
        total_rel_hit = float(
            event_df.loc[event_df["hit"] == 1, "rel_loss_x_duration"].sum()
        )
        E_loss_hit = total_rel_hit / n_hit
    else:
        total_rel_hit = 0.0
        E_loss_hit = float("nan")
    return {
        "sim_id": int(event_df["sim_id"].iloc[0]),
        "scenario": str(event_df["scenario"].iloc[0]),
        "family": str(event_df["family"].iloc[0]),
        "n_events": n_ev,
        "n_nodes": n_nodes,
        "baseline_efficiency": float(event_df["baseline_efficiency"].iloc[0]),
        "L_tilde": L_tilde,
        "L_lcc_tilde": L_lcc,
        "total_rel_loss_x_duration": total_rel,  # sum_i (ΔE/E · T_i)
        "total_rel_loss_x_duration_hit": total_rel_hit,  # sum only over hit
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
    unit_boundary,
) -> tuple[nx.Graph, str, dict]:
    """Return the baseline or cached corridor-expanded graph."""
    if scenario.densify_pct is None:
        return G_base, "base_10km", {}
    graph, meta = load_or_build_densified(
        G_base, unit, scenario.densify_pct, densify_cache, boundary=unit_boundary
    )
    return graph, scenario.network_variant, meta


def _git_commit(project_root: Path) -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


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
    k5km_pkl: Path | None = None,  # retained for the legacy A/B runner
    bootstrap_pool: str = "all",
    outage_year: int | None = None,
    force: bool = False,
    write_events: bool = True,
    epicenter_mode: str = "population",
) -> str:
    """
    Run all Batch-1 scenarios for one analysis unit with shared CRN draws.

    Outputs under out_dir / <unit>/ :
      crn_events.csv.gz          — shared event draws (when write_events=True)
      events_<scenario>.csv.gz   — per-scenario losses (when write_events=True)
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

    county_post_df, flat_lam = load_posterior()
    outages_df = pd.read_csv(
        project_root / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
    )
    county_geom_mcc = load_county_geometry_and_mcc(project_root)
    cov_df = load_coverage_history(project_root)
    county_polys = load_county_polygons(project_root)
    if county_polys.crs is not None:
        county_polys = county_polys.to_crs(4326)
    poly_dict = {row["fips_str"]: row["geometry"] for _, row in county_polys.iterrows()}
    target_fips = set(
        county_post_df.loc[county_post_df["state"].isin(target_states), "fips"]
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    unit_geometry = county_polys[county_polys["fips_str"].isin(target_fips)].geometry
    if unit_geometry.empty:
        raise ValueError(f"No county boundary found for {unit}")
    unit_boundary = (
        unit_geometry.union_all()
        if hasattr(unit_geometry, "union_all")
        else unit_geometry.unary_union
    )
    pop_gdf = (
        require_pop_units(project_root)
        if (epicenter_mode or "population").lower() == "population"
        else None
    )

    # Preload / build graphs + cache E(G)
    # Epicenters always drawn once per sim (population/KDE on G_base); densified
    # graphs only for loss evaluation.
    graph_cache: dict[str, tuple[nx.Graph, float, dict]] = {}
    scenario_variants: dict[str, str] = {}
    for sc in scenarios:
        G_sc, variant, meta = resolve_graph_for_scenario(
            sc,
            G_base=G_base,
            unit=unit,
            densify_cache=densify_cache,
            unit_boundary=unit_boundary,
        )
        scenario_variants[sc.key] = variant
        if variant not in graph_cache:
            E0 = float(nx.global_efficiency(G_sc))
            if not np.isfinite(E0) or E0 <= 0:
                raise ValueError(f"Invalid baseline efficiency for {unit}/{variant}: {E0}")
            graph_cache[variant] = (G_sc, E0, meta)

    if not scenarios:
        return f"empty-scenarios:{unit}"

    crn_rows: list[pd.DataFrame] = []
    metrics_by_sc: dict[str, list[dict]] = {s.key: [] for s in scenarios}
    events_by_sc: dict[str, list[pd.DataFrame]] = {s.key: [] for s in scenarios}
    # Shared removal cache for the unit; keys include graph_id (network variant).
    from collections import OrderedDict

    removal_cache: OrderedDict = OrderedDict()

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
        )
        if events.empty:
            for sc in scenarios:
                variant = scenario_variants[sc.key]
                G_use, E0, _ = graph_cache[variant]
                metrics_by_sc[sc.key].append(
                    {
                        "sim_id": sim_id,
                        "scenario": sc.key,
                        "family": sc.family,
                        "n_events": 0,
                        "n_nodes": G_use.number_of_nodes(),
                        "baseline_efficiency": E0,
                        "L_tilde": 0.0,
                        "L_lcc_tilde": 0.0,
                        "total_rel_loss_x_duration": 0.0,
                        "total_rel_loss_x_duration_hit": 0.0,
                        "mean_pct_eff_loss": np.nan,
                        "mean_lcc_frac": np.nan,
                        "n_disrupted_nodes_mean": 0.0,
                        "n_zero_loss_events": 0,
                        "n_hit_events": 0,
                        "P_hit": np.nan,
                        "L_event": np.nan,
                        "E_loss_given_hit": np.nan,
                        "n_fallback_events": 0,
                        "max_disruption_radius_km": 0.0,
                    }
                )
            continue
        if write_events:
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

        for sc in scenarios:
            variant = scenario_variants[sc.key]
            G_use, E0, _ = graph_cache[variant]
            ev = evaluate_events_on_graph(
                events=events,
                G=G_use,
                E0=E0,
                scenario=sc,
                removal_cache=removal_cache,
                graph_id=variant,
            )
            if write_events:
                events_by_sc[sc.key].append(ev)
            metrics_by_sc[sc.key].append(aggregate_sim_metrics(ev))

        if (sim_id + 1) % 5 == 0 or (sim_id + 1) == n_sims:
            elapsed = time.time() - t0
            print(
                "[Interventions] "
                f"{label} {sim_id+1}/{n_sims} "
                f"({100*(sim_id+1)/n_sims:.0f}%) elapsed {elapsed/60:.1f} min",
                flush=True,
            )

    if not any(metrics_by_sc.values()):
        return f"empty:{unit}"

    if write_events and crn_rows:
        pd.concat(crn_rows, ignore_index=True).to_csv(
            unit_out / "crn_events.csv.gz", index=False, compression="gzip"
        )

    summary_rows: list[dict] = []
    baseline_mean = None
    for sc in scenarios:
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
                "W_mean": float(mdf["total_rel_loss_x_duration"].mean()) if len(mdf) else float("nan"),
                "P_hit_mean": float(mdf["P_hit"].mean()) if len(mdf) else float("nan"),
                "L_event_mean": float(mdf["L_event"].mean()) if len(mdf) else float("nan"),
                "E_loss_given_hit_mean": float(mdf["E_loss_given_hit"].mean()) if len(mdf) else float("nan"),
                "n_events_mean": float(mdf["n_events"].mean()) if len(mdf) else float("nan"),
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

    densify_metas = {}
    for sc in scenarios:
        if sc.densify_pct is not None:
            meta = graph_cache[scenario_variants[sc.key]][2]
            densify_metas[sc.key] = {
                k: meta.get(k)
                for k in (
                    "mode",
                    "pct",
                    "n_target",
                    "n_added",
                    "n_nevi",
                    "n_base",
                    "n_shortfall",
                    "realized_pct",
                )
                if k in meta or k in ("n_nevi", "n_target", "n_added", "n_shortfall")
            }

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    if densify_metas:
        (unit_out / "densify_meta.json").write_text(
            json.dumps(densify_metas, indent=2), encoding="utf-8"
        )
    manifest = {
        "unit": unit,
        "label": label,
        "member_states": target_states,
        "n_sims": n_sims,
        "outage_year": outage_year,
        "bootstrap_pool": bootstrap_pool,
        "epicenter_mode": epicenter_mode,
        "seed_rule": (
            "posterior stream shared by sim_id; unit-keyed independent event-count, "
            "bootstrap, and epicenter streams; all scenarios share unit events"
        ),
        "scenarios": [s.key for s in scenarios],
        "pkl_path": str(pkl_path),
        "densify_cache": str(densify_cache),
        "densify_meta_keys": sorted(densify_metas.keys()),
        "git_commit": _git_commit(project_root),
        "write_events": bool(write_events),
        "removal_cache_entries": int(len(removal_cache)),
    }
    (unit_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return f"ok:{unit}"
