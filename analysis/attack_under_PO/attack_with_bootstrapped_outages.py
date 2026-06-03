#!/usr/bin/env python3
"""
Attack EV charging network using bootstrapped outage events and posterior λ.

Assumptions
-----------
- For a given state, you already have a pickle file with the network:
  data_path: contains a dict with at least:
    - 'network': NetworkX graph of charging stations in that state
      each node has a 'location' attribute: (lat, lon) in degrees.
- You also have a CSV of bootstrapped outage events with impact radius:
  advi_results_2018_2023/bootstrapped_outages_with_radius_sample1.csv
  It must contain at least:
    - 'state'          : state name (e.g., 'California')
    - 'impact_radius_km'
    - 'epicenter_lat'
    - 'epicenter_lon'
    - 'duration' or 'duration_hours'  (outage duration in hours)

For each outage event in the chosen state:
1. Find all nodes within radius r (km) of the epicenter.
2. Remove those nodes (for this event only) and recompute global efficiency.
3. Record:
   - efficiency_before
   - efficiency_after
   - eff_loss          = before - after
   - pct_eff_loss      = eff_loss / before
   - loss_x_duration   = eff_loss * duration_hours

Outputs
-------
- results/{state_name}_attack_events_posterior_lambda.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pickle


def haversine_km(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


def load_network(data_path: Path) -> nx.Graph:
    """Load NetworkX graph from pickle file."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    G = data["network"]
    return G


_DEBUG_COUNTER = 0
_DEBUG_MAX_PRINT = 10
# 正式分析：不放大半径，直接用物理半径
_DEBUG_RADIUS_FACTOR = 1.0


def compute_event_loss(
    G: nx.Graph,
    epicenter: Tuple[float, float],
    radius_km: float,
    eff_before: float | None = None,
) -> Tuple[float, float, float, float]:
    """
    Compute efficiency loss for a single outage event.

    Parameters
    ----------
    G : nx.Graph
    epicenter : (lat, lon)
    radius_km : float
    eff_before : float, optional
        Precomputed baseline efficiency E(G). If None, it will be computed.

    Returns
    -------
    eff_before, eff_after, eff_loss, pct_loss
    """
    if eff_before is None:
        eff_before = nx.global_efficiency(G)

    global _DEBUG_COUNTER

    # 放大半径做调试（正式分析时将 _DEBUG_RADIUS_FACTOR 设为 1.0）
    eff_radius_km = radius_km * _DEBUG_RADIUS_FACTOR

    # Identify affected nodes
    affected = []
    for node in G.nodes:
        node_data = G.nodes[node]
        loc = node_data.get("location")
        # 如果没有 location，就退回使用 (lat, lon)
        if loc is None:
            if "lat" in node_data and "lon" in node_data:
                loc = (node_data["lat"], node_data["lon"])
            else:
                continue
        dist = haversine_km(epicenter, loc)
        if dist <= eff_radius_km:
            affected.append(node)

    if _DEBUG_COUNTER < _DEBUG_MAX_PRINT:
        print(
            f"[DEBUG] E0={eff_before:.6f}, radius_km={radius_km:.2f}, "
            f"affected_nodes={len(affected)}"
        )
        _DEBUG_COUNTER += 1

    if not affected:
        return eff_before, eff_before, 0.0, 0.0

    # Remove affected nodes for this event
    G_temp = G.copy()
    G_temp.remove_nodes_from(affected)

    if G_temp.number_of_nodes() == 0:
        eff_after = 0.0
    else:
        eff_after = nx.global_efficiency(G_temp)

    eff_loss = eff_before - eff_after
    pct_loss = eff_loss / eff_before if eff_before > 0 else 0.0
    return eff_before, eff_after, eff_loss, pct_loss


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Attack EV network using bootstrapped outages and posterior λ."
    )
    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="State name (e.g., 'California').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to state network pickle (contains 'network').",
    )
    parser.add_argument(
        "--outage-csv",
        type=str,
        default="advi_results_2018_2023/bootstrapped_outages_with_radius_sample1.csv",
        help="Bootstrapped outages with radius CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save event-level loss metrics.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    state_name = args.state
    data_path = Path(args.data_path)
    outage_path = (project_root / args.outage_csv) if not Path(args.outage_csv).is_absolute() else Path(args.outage_csv)

    if not data_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {data_path}")
    if not outage_path.exists():
        raise FileNotFoundError(f"Outage CSV not found: {outage_path}")

    print(f"Loading network from: {data_path}")
    G = load_network(data_path)

    print(f"Loading outages from: {outage_path}")
    events = pd.read_csv(outage_path)

    # Filter to chosen state
    events_state = events[events["state"] == state_name].copy()
    if events_state.empty:
        raise ValueError(f"No outage events found for state: {state_name}")

    # Ensure year column exists for yearly aggregation
    if "year" not in events_state.columns:
        if "start_time" in events_state.columns:
            events_state["year"] = pd.to_datetime(events_state["start_time"]).dt.year
        else:
            raise ValueError("Outage CSV must contain 'year' or 'start_time' column.")

    # Columns for epicenter and radius
    if not {"epicenter_lat", "epicenter_lon", "impact_radius_km"}.issubset(
        events_state.columns
    ):
        raise ValueError(
            "Outage CSV must contain 'epicenter_lat', 'epicenter_lon', and 'impact_radius_km'."
        )

    # Duration column
    if "duration_hours" in events_state.columns:
        events_state["duration_hours"] = events_state["duration_hours"].astype(float)
    elif "duration" in events_state.columns:
        events_state["duration_hours"] = events_state["duration"].astype(float)
    else:
        # default to 1 hour if missing
        events_state["duration_hours"] = 1.0

    records = []
    NEG_TOL_SMALL = 1e-8
    NEG_TOL_LARGE = 1e-4
    for idx, row in events_state.iterrows():
        epicenter = (row["epicenter_lat"], row["epicenter_lon"])
        radius_km = float(row["impact_radius_km"])
        duration_h = float(row["duration_hours"])

        eff_before, eff_after, eff_loss, pct_loss = compute_event_loss(
            G, epicenter, radius_km
        )
        # Handle negative losses: small ones as numerical noise, large ones warn then clamp
        if pct_loss < -NEG_TOL_LARGE:
            print(
                f"[WARN] Large negative loss in outage_attack (event {idx}): "
                f"pct_loss={pct_loss:.3e}, eff_before={eff_before:.3e}, eff_after={eff_after:.3e}"
            )
        if pct_loss < 0:
            eff_loss = 0.0
            pct_loss = 0.0
        # Relative loss first, then multiply by duration (matches paper formula)
        rel_loss_x_duration = pct_loss * duration_h
        # Absolute-efficiency version retained for backward compatibility
        loss_x_duration = eff_loss * duration_h

        rec = {
            "event_index": idx,
            "state": state_name,
            "year": int(row["year"]),
            "epicenter_lat": row["epicenter_lat"],
            "epicenter_lon": row["epicenter_lon"],
            "radius_km": radius_km,
            "duration_hours": duration_h,
            "eff_before": eff_before,
            "eff_after": eff_after,
            "eff_loss": eff_loss,
            "pct_eff_loss": pct_loss,
            "loss_x_duration": loss_x_duration,
            "rel_loss_x_duration": rel_loss_x_duration,
        }
        records.append(rec)

    result_df = pd.DataFrame(records)

    # Yearly aggregation: sum of losses per year
    yearly_df = (
        result_df.groupby("year", observed=True)
        .agg(
            n_events=("event_index", "count"),
            total_eff_loss=("eff_loss", "sum"),
            total_loss_x_duration=("loss_x_duration", "sum"),
            # Annual Loss in the paper: sum_i (relative loss_i × duration_i)
            total_rel_loss_x_duration=("rel_loss_x_duration", "sum"),
            mean_pct_eff_loss=("pct_eff_loss", "mean"),
        )
        .reset_index()
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / f"{state_name}_attack_events_posterior_lambda.csv"
    result_df.to_csv(out_path, index=False)

    yearly_path = Path(args.output_dir) / f"{state_name}_attack_yearly_posterior_lambda.csv"
    yearly_df.to_csv(yearly_path, index=False)

    print(f"✓ Saved event-level attack metrics to: {out_path}")
    print(result_df.head())
    print(f"\n✓ Saved yearly aggregated losses to: {yearly_path}")
    print(yearly_df.head())

    # Optional: state-level node-normalized vulnerability for this single run
    n_nodes = G.number_of_nodes() if G is not None else 0
    if n_nodes > 0:
        annual_loss_mean = yearly_df["total_rel_loss_x_duration"].mean()
        node_norm_loss = annual_loss_mean / n_nodes
        print(
            f"\nNode-normalized vulnerability (single-run, relative-loss-based): "
            f"{node_norm_loss:.6e} (n_nodes={n_nodes})"
        )


if __name__ == "__main__":
    main()

