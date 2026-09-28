#!/usr/bin/env python3
"""
Monte Carlo outage-attack simulation using posterior λ and EV networks.

Workflow for ONE state (repeat N times):
1. From ADVI posterior, draw one realization of county-level λ_c.
2. Aggregate to state λ_s, draw n_s ~ Poisson(λ_s).
3. Bootstrap n_s historical outages for that state (county chosen ∝ λ_c).
4. For each outage event:
   - Use impacted customers + coverage + MCC + area to compute radius r.
   - Use epicenter + r to blackout stations in that state network.
   - Compute efficiency loss and loss × duration.
5. Aggregate per year, record metrics with sim_id.

Outputs
-------
- results/<STATE>_attack_yearly_mc.csv
  Columns: sim_id, year, n_events, total_eff_loss, total_loss_x_duration, mean_pct_eff_loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import geopandas as gpd
from shapely.geometry import Point

from sample_lambda_from_posterior import (
    build_raw_lambda_draw,
    load_county_estimates,
    load_posterior,
    draw_lambda_realization,
    aggregate_state_lambda,
    draw_state_event_counts,
    bootstrap_state_outages,
)
from compute_impact_radius import (
    load_county_geometry_and_mcc,
    load_coverage_history,
    resolve_county_shapefile,
    assert_county_geoids_match_eagle_i,
    STATE_NAME_TO_ABBR,
)
from attack_with_bootstrapped_outages import (
    _affected_nodes_in_radius,
    compute_event_loss,
    compute_event_loss_with_capacity,
)
from select_epicenter_by_stations import choose_station_epicenter_in_county
from select_epicenter_by_population import (
    choose_epicenter_population_mode,
    require_pop_units,
)


def haversine_km(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def load_network(data_path: Path) -> nx.Graph:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    G = data["network"]
    return G


def load_county_polygons(project_root: Path) -> gpd.GeoDataFrame:
    """Load county polygons (CONUS + DC) with fips_str (EAGLE-I FIPS vintage)."""
    shp_path = resolve_county_shapefile(project_root)
    gdf = gpd.read_file(shp_path)
    if "STATEFP" in gdf.columns:
        non_continental = {"02", "15", "72", "78", "60", "66", "69"}
        gdf = gdf[~gdf["STATEFP"].astype(str).isin(non_continental)].copy()

    if "GEOID" not in gdf.columns:
        raise ValueError("Shapefile must contain GEOID column for county FIPS.")

    gdf["fips_str"] = gdf["GEOID"].astype(str).str.zfill(5)
    assert_county_geoids_match_eagle_i(gdf["fips_str"])
    return gdf[["fips_str", "STATEFP", "geometry"]]


def sample_point_in_polygon(poly, rng: np.random.Generator) -> Point:
    """Uniform-ish random point inside polygon via rejection sampling."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(100):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            return p
    # Fallback to representative point if rejection fails
    return poly.representative_point()


def compute_radius_for_state_events(
    events_state: pd.DataFrame,
    county_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    state_name: str,
) -> pd.DataFrame:
    """
    Given bootstrapped events for a single state, append impact_radius_km.

    events_state must contain:
      - 'fips' or 'sim_county_fips'
      - 'Nd' (impacted customers; we'll derive if missing)
      - 'year'
    """
    df = events_state.copy()

    # FIPS
    if "sim_county_fips" in df.columns:
        df["fips_str"] = df["sim_county_fips"].astype(str).str.zfill(5)
    elif "fips" in df.columns:
        df["fips_str"] = df["fips"].astype(int).astype(str).str.zfill(5)
    else:
        raise ValueError("events_state must contain 'sim_county_fips' or 'fips'.")

    # Impacted customers Nd
    if "Nd" not in df.columns:
        if "affected_customers" in df.columns:
            df["Nd"] = df["affected_customers"].astype(float)
        elif "mean_customers" in df.columns:
            df["Nd"] = df["mean_customers"].astype(float)
        else:
            raise ValueError(
                "events_state must contain 'Nd' or 'affected_customers' or 'mean_customers'."
            )

    # Merge county info (MCC, area_km2, rho)
    df = df.merge(county_df, on="fips_str", how="left")

    # Coverage
    abbr = STATE_NAME_TO_ABBR.get(state_name)
    if abbr is None:
        raise ValueError(f"No state abbreviation mapping for: {state_name}")

    cov_state = cov_df[cov_df["state_abbr"] == abbr].copy()
    df = df.merge(
        cov_state,
        left_on=["year"],
        right_on=["year_num"],
        how="left",
    )

    df["theta_s"] = df["theta_s"].fillna(1.0).clip(lower=1e-3, upper=1.0)

    # Effective impacted customers: min(Nd/theta_s, MCC)
    Nd_eff = (df["Nd"] / df["theta_s"]).clip(lower=0.0)
    Nd_eff = np.minimum(Nd_eff, df["MCC"])

    rho = df["rho"].clip(lower=1e-6)
    df["impact_radius_km"] = np.sqrt(Nd_eff / (rho * np.pi))
    return df


def _progress_dir() -> Path:
    import os

    env = os.environ.get("MC_PROGRESS_DIR")
    if env:
        d = Path(env)
    else:
        d = Path(__file__).resolve().parents[2] / "results_mc_10km_panel_2018_2026" / "_progress"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_mc_progress(
    state_name: str,
    *,
    sims_done: int,
    n_sims: int,
    elapsed_min: float,
    remaining_min: float,
) -> None:
    import json

    safe = state_name.replace(" ", "_")
    payload = {
        "state": state_name,
        "sims_done": sims_done,
        "n_sims": n_sims,
        "pct": round(100.0 * sims_done / n_sims, 1),
        "elapsed_min": round(elapsed_min, 2),
        "remaining_min": round(remaining_min, 2),
    }
    path = _progress_dir() / f"{safe}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_monte_carlo_for_state(
    state_name: str,
    data_path: Path,
    n_sims: int = 100,
    *,
    member_states: list[str] | None = None,
    capacity_weighted: bool = False,
    lambda_mode: str = "posterior",
    bootstrap_pool: str = "all",
    epicenter_mode: str = "population",
    kde_sigma_km: float = 3.0,
) -> pd.DataFrame:
    """
    epicenter_mode
    --------------
    population (default): census-tract population weighted (network-independent).
        Requires data/processed/pop_units_epicenter.gpkg.
        Zero-loss events are expected (real outages often miss EV stations).
    station: legacy on-node epicenter (appendix ablation only; inflates P(hit)).
    """
    del kde_sigma_km  # legacy kw; station-KDE removed
    project_root = Path(__file__).resolve().parents[2]
    _write_mc_progress(state_name, sims_done=0, n_sims=n_sims, elapsed_min=0.0, remaining_min=0.0)

    # Load static inputs
    lambda_mode = lambda_mode.lower().strip()
    if lambda_mode not in ("posterior", "raw"):
        raise ValueError(f"lambda_mode must be 'posterior' or 'raw', got {lambda_mode!r}")
    bootstrap_pool = bootstrap_pool.lower().strip()
    if bootstrap_pool not in ("all", "severe_top10"):
        raise ValueError(
            f"bootstrap_pool must be 'all' or 'severe_top10', got {bootstrap_pool!r}"
        )
    epicenter_mode = epicenter_mode.lower().strip()
    if epicenter_mode not in ("population", "station"):
        raise ValueError(
            f"epicenter_mode must be 'population' or 'station', got {epicenter_mode!r}"
        )
    county_post_df = load_county_estimates()
    flat_lam = None
    if lambda_mode == "posterior":
        county_post_df, flat_lam = load_posterior()
    outages_df = pd.read_csv(
        project_root
        / "notebooks"
        / "PO_data_cleaning"
        / "cleaned_outages_2018_2023.csv"
    )
    county_geom_mcc = load_county_geometry_and_mcc(project_root)
    cov_df = load_coverage_history(project_root)
    county_polys = load_county_polygons(project_root)
    G = load_network(data_path)
    pop_gdf = require_pop_units(project_root) if epicenter_mode == "population" else None
    # Precompute baseline efficiency once to reuse across events
    try:
        E0 = nx.global_efficiency(G)
    except Exception:
        E0 = 0.0

    # Mapping from fips_str -> polygon (target state only)
    target_states = list(member_states) if member_states else [state_name]
    for s in target_states:
        if STATE_NAME_TO_ABBR.get(s) is None:
            raise ValueError(f"No state abbreviation mapping for: {s}")

    state_polys = county_polys.copy()
    poly_dict = {row["fips_str"]: row["geometry"] for _, row in state_polys.iterrows()}

    all_yearly: list[pd.DataFrame] = []
    all_sim_metrics: list[dict] = []

    import time

    t_start = time.time()
    n_nodes = G.number_of_nodes() if G is not None else 0

    for sim_id in range(n_sims):
        # 1) County λ: posterior draw or fixed raw MLE (n_events/n_years)
        if lambda_mode == "posterior":
            df_draw = draw_lambda_realization(
                county_post_df, flat_lam, random_state=sim_id
            )
        else:
            df_draw = build_raw_lambda_draw(county_post_df)
        state_df = aggregate_state_lambda(df_draw)
        state_df = draw_state_event_counts(state_df)

        # 2) Bootstrap outages (all states), then filter our target state
        boot_all = bootstrap_state_outages(
            state_df,
            df_draw,
            outages_df,
            random_state=sim_id,
            event_pool=bootstrap_pool,
        )
        boot_state = boot_all[boot_all["sim_state"].isin(target_states)].copy()
        if boot_state.empty:
            continue

        # year column should exist from cleaned_outages
        if "year" not in boot_state.columns:
            if "start_time" in boot_state.columns:
                boot_state["year"] = pd.to_datetime(
                    boot_state["start_time"]
                ).dt.year
            else:
                raise ValueError("bootstrapped outages must contain 'year' or 'start_time'.")

        # 3) Compute radius for each event
        # Coverage / radius uses per-event state name (still one of the member states)
        boot_chunks = []
        for ev_state in boot_state["sim_state"].unique():
            chunk = boot_state[boot_state["sim_state"] == ev_state]
            boot_chunks.append(
                compute_radius_for_state_events(
                    chunk, county_geom_mcc, cov_df, str(ev_state)
                )
            )
        boot_state = pd.concat(boot_chunks, axis=0, ignore_index=True)

        # 3b) Assign epicenters per event (population-weighted by default).
        # Zero-loss events are expected under population mode — real outages
        # often miss EV stations. Epicenter coords feed L_event = P(hit)×E[loss|hit].
        rng = np.random.default_rng(sim_id)
        epic_lats = []
        epic_lons = []
        for _, row in boot_state.iterrows():
            fips_str = str(row["fips_str"]).zfill(5)
            poly = poly_dict.get(fips_str)

            if epicenter_mode == "station":
                station_epic = choose_station_epicenter_in_county(
                    G,
                    county_fips=fips_str,
                    county_attr="county_fips",
                    weight_attr=None,
                    rng=rng,
                )
                if station_epic is not None:
                    epic_lats.append(station_epic[0])
                    epic_lons.append(station_epic[1])
                    continue
                if poly is None:
                    epic_lats.append(np.nan)
                    epic_lons.append(np.nan)
                    continue
                p = sample_point_in_polygon(poly, rng)
                epic_lats.append(p.y)
                epic_lons.append(p.x)
                continue

            # population mode: census-tract only (no station-KDE)
            epic, _method = choose_epicenter_population_mode(
                county_fips=fips_str,
                poly=poly,
                rng=rng,
                pop_gdf=pop_gdf,
            )
            if epic is None:
                epic_lats.append(np.nan)
                epic_lons.append(np.nan)
            else:
                epic_lats.append(epic[0])
                epic_lons.append(epic[1])

        boot_state["epicenter_lat"] = epic_lats
        boot_state["epicenter_lon"] = epic_lons

        # Drop events without epicenter
        boot_state = boot_state.dropna(subset=["epicenter_lat", "epicenter_lon"])
        if boot_state.empty:
            continue

        # 4) Attack network per event.
        # Miss (no node in R_c) is a valid outcome: record hit=0, ΔE/E=0.
        # Do NOT skip the event and do NOT redraw the epicenter.
        records = []
        event_ch: list[float] = []
        event_rel: list[float] = []
        for _, row in boot_state.iterrows():
            radius_km = float(row["impact_radius_km"])
            if "duration_hours" in row and pd.notna(row.get("duration_hours")):
                duration_h = float(row["duration_hours"])
            elif "duration_min" in row and pd.notna(row.get("duration_min")):
                duration_h = float(row["duration_min"]) / 60.0
            else:
                duration_h = 1.0
            fips_str = row["fips_str"]
            epicenter = (float(row["epicenter_lat"]), float(row["epicenter_lon"]))

            cap_disrupted = 0.0
            if capacity_weighted:
                (
                    eff_before,
                    eff_after,
                    eff_loss,
                    pct_loss,
                    _n_aff,
                    cap_disrupted,
                    lcc_frac_val,
                ) = compute_event_loss_with_capacity(
                    G, epicenter, radius_km, eff_before=E0
                )
            else:
                (
                    eff_before,
                    eff_after,
                    eff_loss,
                    pct_loss,
                    lcc_frac_val,
                ) = compute_event_loss(
                    G, epicenter, radius_km, eff_before=E0
                )

            # Numerical noise only — never redraw epicenter on miss/zero loss
            if pct_loss is not None and pct_loss < 0:
                eff_loss = 0.0
                pct_loss = 0.0

            # L_i = (E(G)-E(G\\S))/E(G) × T_i  (paper); optional × C_{S_i}
            rel_loss_x_duration = pct_loss * duration_h
            if capacity_weighted:
                rel_loss_x_duration *= cap_disrupted
            loss_x_duration = eff_loss * duration_h if eff_loss is not None else 0.0

            lcc_deficit = max(0.0, 1.0 - float(lcc_frac_val))
            rel_lcc_loss_x_duration = lcc_deficit * duration_h

            epicenter_final = epicenter
            n_disrupted = len(
                _affected_nodes_in_radius(G, epicenter_final, radius_km)
            )
            hit = int(1 if n_disrupted > 0 else 0)
            # Miss ⇒ hit=0 and ΔE/E=0 by construction (keep in P(hit) denom / L mean)
            if hit == 0:
                pct_loss = 0.0
                eff_loss = 0.0
                rel_loss_x_duration = 0.0
                loss_x_duration = 0.0

            nd = float(row.get("Nd", row.get("affected_customers", 0.0)) or 0.0)
            customer_hours = nd * duration_h
            event_ch.append(customer_hours)
            event_rel.append(rel_loss_x_duration)

            records.append(
                {
                    "sim_id": sim_id,
                    "seed": int(sim_id),
                    "year": int(row["year"]),
                    "fips_str": str(fips_str).zfill(5),
                    "epicenter_lat": float(epicenter_final[0]),
                    "epicenter_lon": float(epicenter_final[1]),
                    "hit": hit,
                    "n_disrupted_nodes": int(n_disrupted),  # |S_i|
                    "eff_loss": eff_loss if eff_loss is not None else 0.0,
                    "pct_eff_loss": pct_loss if pct_loss is not None else 0.0,  # ΔE/E
                    "duration_hours": float(duration_h),  # T_i
                    "lcc_frac": lcc_frac_val,
                    "lcc_deficit": lcc_deficit,
                    "loss_x_duration": loss_x_duration,
                    "rel_loss_x_duration": rel_loss_x_duration,
                    "rel_lcc_loss_x_duration": rel_lcc_loss_x_duration,
                    "capacity_disrupted": cap_disrupted,
                    "impact_radius_km": radius_km,
                    "customer_hours": customer_hours,
                    "used_state_pool_fallback": bool(
                        row.get("used_state_pool_fallback", False)
                    ),
                }
            )

        if not records:
            continue

        df_events = pd.DataFrame(records)

        # Pooled over years: treat 2018–2022 outages as one representative "year"
        yearly = (
            df_events.assign(year_pooled="pooled")
            .groupby("year_pooled", observed=True)
            .agg(
                n_events=("eff_loss", "count"),
                total_eff_loss=("eff_loss", "sum"),
                total_loss_x_duration=("loss_x_duration", "sum"),
                # Annual Loss in the paper: sum_i (relative loss_i × duration_i)
                total_rel_loss_x_duration=("rel_loss_x_duration", "sum"),
                total_rel_lcc_loss_x_duration=("rel_lcc_loss_x_duration", "sum"),
                mean_pct_eff_loss=("pct_eff_loss", "mean"),
                mean_lcc_frac=("lcc_frac", "mean"),
                total_capacity_disrupted=("capacity_disrupted", "sum"),
            )
            .reset_index()
            .rename(columns={"year_pooled": "year"})
        )
        yearly["sim_id"] = sim_id
        yearly["baseline_efficiency"] = E0
        yearly["capacity_weighted"] = capacity_weighted
        yearly["n_nodes"] = n_nodes
        L_tilde = 0.0
        L_lcc_tilde = 0.0
        if n_nodes > 0:
            L_tilde = float(yearly["total_rel_loss_x_duration"].iloc[0] / n_nodes)
            L_lcc_tilde = float(yearly["total_rel_lcc_loss_x_duration"].iloc[0] / n_nodes)
            yearly["node_normalized_loss"] = L_tilde
            yearly["node_normalized_lcc_loss"] = L_lcc_tilde

        # Sensitivity: normal = below-median severity events; severe = top quartile
        L_normal = L_severe = 0.0
        if n_nodes > 0 and event_ch:
            ch_arr = np.asarray(event_ch, dtype=float)
            rel_arr = np.asarray(event_rel, dtype=float)
            med = float(np.median(ch_arr))
            p75 = float(np.percentile(ch_arr, 75))
            L_normal = float(rel_arr[ch_arr <= med].sum() / n_nodes)
            L_severe = float(rel_arr[ch_arr >= p75].sum() / n_nodes)

        n_ev = int(df_events.shape[0])
        # P(hit) uses hit flag (misses stay in the denominator); E[loss|hit] excludes them
        n_hit = int(df_events["hit"].sum()) if "hit" in df_events.columns else 0
        n_zero = n_ev - n_hit
        p_hit = float(n_hit / n_ev) if n_ev > 0 else float("nan")
        total_rel = float(df_events["rel_loss_x_duration"].sum())
        L_event = total_rel / n_ev if n_ev > 0 else float("nan")
        # Only hit events enter the conditional mean (misses have loss 0 anyway)
        if n_hit > 0 and "hit" in df_events.columns:
            E_loss_hit = float(
                df_events.loc[df_events["hit"] == 1, "rel_loss_x_duration"].sum() / n_hit
            )
        else:
            E_loss_hit = float("nan")

        all_sim_metrics.append(
            {
                "sim_id": sim_id,
                "L_tilde": L_tilde,
                "L_lcc_tilde": L_lcc_tilde,
                "mean_lcc_frac": float(df_events["lcc_frac"].mean()),
                "n_events": n_ev,
                "n_disrupted_nodes_mean": float(df_events["n_disrupted_nodes"].mean()),
                "outage_severity_mean": float(df_events["customer_hours"].mean()),
                "n_zero_loss_events": n_zero,
                "n_hit_events": n_hit,
                "P_hit": p_hit,
                "L_event": L_event,
                "E_loss_given_hit": E_loss_hit,
                "max_disruption_radius_km": float(df_events["impact_radius_km"].max()),
                "n_fallback_events": int(df_events["used_state_pool_fallback"].sum()),
                "L_tilde_normal": L_normal,
                "L_tilde_severe": L_severe,
                "epicenter_mode": epicenter_mode,
            }
        )
        all_yearly.append(yearly)

        # 进度条：每 5 次或最后一次（并行时各 worker 各打一行）
        if (sim_id + 1) % 5 == 0 or (sim_id + 1) == n_sims:
            elapsed = time.time() - t_start
            sims_done = sim_id + 1
            avg_per_sim = elapsed / sims_done
            remaining = avg_per_sim * (n_sims - sims_done)
            pct = 100.0 * sims_done / n_sims
            bar_w = 30
            filled = int(bar_w * sims_done / n_sims)
            bar = "#" * filled + "-" * (bar_w - filled)
            msg = (
                f"[MC] {state_name} |{bar}| {sims_done}/{n_sims} ({pct:.0f}%) "
                f"elapsed {elapsed/60:.1f} min, est. remaining {remaining/60:.1f} min"
            )
            print(msg, flush=True)
            _write_mc_progress(
                state_name,
                sims_done=sims_done,
                n_sims=n_sims,
                elapsed_min=elapsed / 60,
                remaining_min=remaining / 60,
            )

    if not all_yearly:
        return pd.DataFrame(), pd.DataFrame()

    yearly_out = pd.concat(all_yearly, axis=0, ignore_index=True)
    sim_out = pd.DataFrame(all_sim_metrics)
    return yearly_out, sim_out


def main() -> None:
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser(
        description="Monte Carlo outage-attack simulation using posterior λ."
    )
    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="State or region label for outputs (e.g. 'Chesapeake region').",
    )
    parser.add_argument(
        "--member-states",
        type=str,
        default="",
        help="Comma-separated full state names for merged regions (e.g. 'Delaware,Maryland').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to state network pickle (from build_network_pickle_only.py).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save Monte Carlo yearly losses.",
    )
    parser.add_argument(
        "--capacity-weighted",
        action="store_true",
        help="Optional: multiply each event by disrupted capacity C_S (default: pure topology)",
    )
    parser.add_argument(
        "--epicenter-mode",
        type=str,
        default="population",
        choices=("population", "station"),
        help="population (default): census-tract pop; station: legacy appendix ablation.",
    )

    args = parser.parse_args()

    state_name = args.state
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {data_path}")

    members = [s.strip() for s in args.member_states.split(",") if s.strip()] or None
    yearly_mc, _sim_mc = run_monte_carlo_for_state(
        state_name=state_name,
        data_path=data_path,
        n_sims=args.n_sims,
        member_states=members,
        capacity_weighted=args.capacity_weighted,
        epicenter_mode=args.epicenter_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = (
        Path(args.output_dir) / f"{state_name}_attack_yearly_mc_posterior_lambda.csv"
    )
    yearly_mc.to_csv(out_path, index=False)

    print(f"[OK] Saved Monte Carlo yearly losses to: {out_path}")
    print(yearly_mc.head())

    # Monte Carlo average of Annual Loss (relative-loss-based) and node-normalized metric
    if not yearly_mc.empty and "total_rel_loss_x_duration" in yearly_mc.columns:
        mean_annual_loss = (
            yearly_mc.groupby("year", observed=True)["total_rel_loss_x_duration"]
            .mean()
            .iloc[0]
        )

        # Load network to get |V| and baseline efficiency
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        G = data["network"]
        n_nodes = G.number_of_nodes() if G is not None else 0
        try:
            E0 = nx.global_efficiency(G)
        except Exception:
            E0 = float("nan")

        node_norm_loss = mean_annual_loss / n_nodes if n_nodes > 0 else float("nan")

        summary_path = (
            Path(args.output_dir)
            / f"{state_name}_attack_mc_node_normalized_posterior_lambda.csv"
        )
        summary_df = pd.DataFrame(
            {
                "state": [state_name],
                "mean_annual_loss_relxduration": [mean_annual_loss],
                "n_nodes": [n_nodes],
                "node_normalized_loss": [node_norm_loss],
                "baseline_efficiency": [E0],
                "capacity_weighted": [args.capacity_weighted],
                "n_sims": [args.n_sims],
            }
        )
        summary_df.to_csv(summary_path, index=False)

        print(
            f"\nMC-averaged Annual Loss (relative-loss × duration): {mean_annual_loss:.6e}"
        )
        print(
            f"Node-normalized vulnerability: {node_norm_loss:.6e}  "
            f"(saved to {summary_path})"
        )


if __name__ == "__main__":
    main()

