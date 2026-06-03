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
    load_posterior,
    draw_lambda_realization,
    aggregate_state_lambda,
    draw_state_event_counts,
    bootstrap_state_outages,
)
from compute_impact_radius import load_county_geometry_and_mcc, load_coverage_history, STATE_NAME_TO_ABBR
from attack_with_bootstrapped_outages import compute_event_loss
from select_epicenter_by_stations import choose_station_epicenter_in_county


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
    """Load county polygons (CONUS + DC) with fips_str."""
    ev_root = project_root.parent
    shp_path = ev_root / "tl_2021_us_county" / "tl_2021_us_county.shp"
    if not shp_path.exists():
        raise FileNotFoundError(f"County shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if "STATEFP" in gdf.columns:
        non_continental = {"02", "15", "72", "78", "60", "66", "69"}
        gdf = gdf[~gdf["STATEFP"].astype(str).isin(non_continental)].copy()

    if "GEOID" not in gdf.columns:
        raise ValueError("Shapefile must contain GEOID column for county FIPS.")

    gdf["fips_str"] = gdf["GEOID"].astype(str).str.zfill(5)
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


def run_monte_carlo_for_state(
    state_name: str,
    data_path: Path,
    n_sims: int = 100,
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]

    # Load static inputs
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
    # Precompute baseline efficiency once to reuse across events
    try:
        E0 = nx.global_efficiency(G)
    except Exception:
        E0 = 0.0

    # Mapping from fips_str -> polygon (target state only)
    abbr = STATE_NAME_TO_ABBR.get(state_name)
    if abbr is None:
        raise ValueError(f"No state abbreviation mapping for: {state_name}")
    # STATEFP is numeric string; map abbr via standard mapping if necessary
    state_polys = county_polys.copy()

    poly_dict = {row["fips_str"]: row["geometry"] for _, row in state_polys.iterrows()}

    all_yearly: list[pd.DataFrame] = []

    import time

    t_start = time.time()

    for sim_id in range(n_sims):
        # 1) Draw λ realization
        df_draw = draw_lambda_realization(county_post_df, flat_lam)
        state_df = aggregate_state_lambda(df_draw)
        state_df = draw_state_event_counts(state_df)

        # 2) Bootstrap outages (all states), then filter our target state
        boot_all = bootstrap_state_outages(
            state_df, df_draw, outages_df, random_state=sim_id
        )
        boot_state = boot_all[boot_all["sim_state"] == state_name].copy()
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
        boot_state = compute_radius_for_state_events(
            boot_state, county_geom_mcc, cov_df, state_name
        )

        # 3b) Assign epicenters per event.
        # 优先在该县的站点中随机选一个 station 作为 epicenter；
        # 若该县没有任何站点（或缺少坐标），则退回到县 polygon 内均匀采样。
        rng = np.random.default_rng(sim_id)
        epic_lats = []
        epic_lons = []
        for _, row in boot_state.iterrows():
            fips_str = row["fips_str"]

            # 尝试：在该县的 station 中选 epicenter
            station_epic = choose_station_epicenter_in_county(
                G,
                county_fips=fips_str,
                county_attr="county_fips",
                weight_attr=None,
                rng=rng,
            )
            if station_epic is not None:
                lat, lon = station_epic
                epic_lats.append(lat)
                epic_lons.append(lon)
                continue

            # 如果该县没有匹配的站点，退回到 county polygon 内均匀采样
            poly = poly_dict.get(fips_str)
            if poly is None:
                epic_lats.append(np.nan)
                epic_lons.append(np.nan)
                continue
            p = sample_point_in_polygon(poly, rng)
            epic_lats.append(p.y)
            epic_lons.append(p.x)

        boot_state["epicenter_lat"] = epic_lats
        boot_state["epicenter_lon"] = epic_lons

        # Drop events without epicenter
        boot_state = boot_state.dropna(subset=["epicenter_lat", "epicenter_lon"])
        if boot_state.empty:
            continue

        # 4) Attack network per event
        records = []
        NEG_TOL_SMALL = 1e-8
        NEG_TOL_LARGE = 1e-4
        for _, row in boot_state.iterrows():
            radius_km = float(row["impact_radius_km"])
            duration_h = float(row.get("duration_hours", 1.0))
            fips_str = row["fips_str"]
            poly = poly_dict.get(fips_str)

            pct_loss = None
            eff_loss = None
            eff_before = None
            eff_after = None

            max_tries = 3
            for attempt in range(max_tries):
                epicenter = (row["epicenter_lat"], row["epicenter_lon"])

                eff_before, eff_after, eff_loss, pct_loss = compute_event_loss(
                    G, epicenter, radius_km, eff_before=E0
                )

                if pct_loss is not None and pct_loss < -NEG_TOL_LARGE:
                    # Large negative loss: warn and, if possible, resample epicenter within county
                    print(
                        f"[WARN] Large negative loss in MC (sim {sim_id}, fips {fips_str}, "
                        f"attempt {attempt}, pct_loss={pct_loss:.3e}). Resampling epicenter."
                    )
                    if poly is not None:
                        p_new = sample_point_in_polygon(poly, rng)
                        row["epicenter_lat"] = p_new.y
                        row["epicenter_lon"] = p_new.x
                        continue
                # Either loss is non-negative (good) or only mildly negative (treat as noise)
                break

            # After retries, clamp small negatives to zero so metrics不会变负
            if pct_loss is not None and pct_loss < 0:
                eff_loss = 0.0
                pct_loss = 0.0

            # Paper formula: L_i = (relative loss) × duration
            rel_loss_x_duration = pct_loss * duration_h
            # Keep absolute-efficiency × duration for backward compatibility
            loss_x_duration = eff_loss * duration_h if eff_loss is not None else 0.0

            records.append(
                {
                    "sim_id": sim_id,
                    "year": int(row["year"]),
                    "eff_loss": eff_loss if eff_loss is not None else 0.0,
                    "pct_eff_loss": pct_loss if pct_loss is not None else 0.0,
                    "loss_x_duration": loss_x_duration,
                    "rel_loss_x_duration": rel_loss_x_duration,
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
                mean_pct_eff_loss=("pct_eff_loss", "mean"),
            )
            .reset_index()
            .rename(columns={"year_pooled": "year"})
        )
        yearly["sim_id"] = sim_id
        # Same baseline efficiency E(G) and |V| for all sims of this state
        yearly["baseline_efficiency"] = E0
        n_nodes = G.number_of_nodes() if G is not None else 0
        if n_nodes > 0:
            # Node-normalized Annual Loss for THIS sim (already MC-iteration specific)
            yearly["node_normalized_loss"] = (
                yearly["total_rel_loss_x_duration"] / n_nodes
            )
        all_yearly.append(yearly)

        # 简单进度 & 剩余时间估计（每 10 次或最后一次打印）
        if (sim_id + 1) % 10 == 0 or (sim_id + 1) == n_sims:
            elapsed = time.time() - t_start
            sims_done = sim_id + 1
            avg_per_sim = elapsed / sims_done
            remaining = avg_per_sim * (n_sims - sims_done)
            print(
                f"[MC] {sims_done}/{n_sims} sims done "
                f"(elapsed {elapsed/60:.1f} min, est. remaining {remaining/60:.1f} min)"
            )

    if not all_yearly:
        return pd.DataFrame()

    return pd.concat(all_yearly, axis=0, ignore_index=True)


def main() -> None:
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser(
        description="Monte Carlo outage-attack simulation using posterior λ."
    )
    parser.add_argument("--state", type=str, required=True, help="State name.")
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

    args = parser.parse_args()

    state_name = args.state
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {data_path}")

    yearly_mc = run_monte_carlo_for_state(
        state_name=state_name,
        data_path=data_path,
        n_sims=args.n_sims,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = (
        Path(args.output_dir) / f"{state_name}_attack_yearly_mc_posterior_lambda.csv"
    )
    yearly_mc.to_csv(out_path, index=False)

    print(f"✓ Saved Monte Carlo yearly losses to: {out_path}")
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

