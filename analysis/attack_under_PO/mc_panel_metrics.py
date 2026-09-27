#!/usr/bin/env python3
"""Aggregate MC outputs for panel metrics (residual / sensitivity / diagnostics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def state_posterior_lambda_mean(
    member_states: list[str],
    project_root: Path,
) -> float:
    """Sum of county posterior mean λ (events/year) over all counties in member states."""
    csv_path = project_root / "advi_results_2018_2023" / "county_estimates_advi.csv"
    if not csv_path.is_file():
        return float("nan")
    df = pd.read_csv(csv_path)
    if "lambda_mean" not in df.columns or "state" not in df.columns:
        return float("nan")
    sub = df[df["state"].isin(member_states)]
    if sub.empty:
        return float("nan")
    return float(sub["lambda_mean"].sum())


def state_raw_lambda_mean(
    member_states: list[str],
    project_root: Path,
) -> float:
    """Sum of county raw λ (n_events/n_years) over member states."""
    csv_path = project_root / "advi_results_2018_2023" / "county_estimates_advi.csv"
    if not csv_path.is_file():
        return float("nan")
    df = pd.read_csv(csv_path)
    if "lambda_raw" not in df.columns or "state" not in df.columns:
        return float("nan")
    sub = df[df["state"].isin(member_states)]
    if sub.empty:
        return float("nan")
    return float(sub["lambda_raw"].sum())


def compute_network_features(G: nx.Graph) -> dict[str, float]:
    """Topology + capacity features for residual analysis."""
    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "avg_degree": float("nan"),
            "density": float("nan"),
            "avg_clustering": float("nan"),
            "total_capacity": 0.0,
            "algebraic_connectivity": float("nan"),
            "diameter": float("nan"),
        }

    caps = [float(G.nodes[v].get("capacity", 0.0) or 0.0) for v in G.nodes]
    und = G.to_undirected() if G.is_directed() else G

    avg_degree = float(2 * m / n) if n else float("nan")
    density = float(nx.density(und))

    try:
        avg_clustering = float(nx.average_clustering(und))
    except Exception:
        avg_clustering = float("nan")

    try:
        if nx.is_connected(und):
            diameter = float(nx.diameter(und))
            alg_conn = float(nx.algebraic_connectivity(und))
        else:
            largest = max(nx.connected_components(und), key=len)
            sub = und.subgraph(largest).copy()
            diameter = float(nx.diameter(sub))
            alg_conn = float(nx.algebraic_connectivity(sub))
    except Exception:
        diameter = float("nan")
        alg_conn = float("nan")

    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_degree,
        "density": density,
        "avg_clustering": avg_clustering,
        "total_capacity": float(sum(caps)),
        "algebraic_connectivity": alg_conn,
        "diameter": diameter,
    }


def aggregate_sim_metrics_to_row(
    sim_df: pd.DataFrame,
    *,
    state: str,
    year: int,
    unit: str,
    baseline_efficiency: float,
    network_features: dict[str, Any],
    outage_lambda: float,
    n_sims: int,
    capacity_weighted: bool,
    network_pkl: str,
) -> dict[str, Any]:
    """One state×year row from per-simulation metrics."""
    if sim_df.empty:
        return {}

    l = sim_df["L_tilde"].astype(float)
    l_lcc = sim_df["L_lcc_tilde"].astype(float) if "L_lcc_tilde" in sim_df.columns else None
    row: dict[str, Any] = {
        "state": state,
        "year": year,
        "unit": unit,
        "n_sims": n_sims,
        "capacity_weighted": capacity_weighted,
        "network_pkl": network_pkl,
        "baseline_efficiency": baseline_efficiency,
        "outage_lambda": outage_lambda,
        # Required
        "n_nodes": network_features.get("n_nodes"),
        "n_edges": network_features.get("n_edges"),
        "L_tilde_mean": float(l.mean()),
        "L_tilde_std": float(l.std(ddof=1)) if len(l) > 1 else 0.0,
        "L_lcc_tilde_mean": float(l_lcc.mean()) if l_lcc is not None else float("nan"),
        "L_lcc_tilde_std": (
            float(l_lcc.std(ddof=1)) if l_lcc is not None and len(l_lcc) > 1 else 0.0
        ),
        "mean_lcc_frac": (
            float(sim_df["mean_lcc_frac"].mean())
            if "mean_lcc_frac" in sim_df.columns
            else float("nan")
        ),
        "n_events_mean": float(sim_df["n_events"].mean()),
        "n_disrupted_nodes_mean": float(sim_df["n_disrupted_nodes_mean"].mean()),
        # Residual features
        "outage_severity_mean": float(sim_df["outage_severity_mean"].mean()),
        "avg_degree": network_features.get("avg_degree"),
        "density": network_features.get("density"),
        "avg_clustering": network_features.get("avg_clustering"),
        "total_capacity": network_features.get("total_capacity"),
        "algebraic_connectivity": network_features.get("algebraic_connectivity"),
        "diameter": network_features.get("diameter"),
        # Sensitivity
        "L_tilde_normal": float(sim_df["L_tilde_normal"].mean()),
        "L_tilde_severe": float(sim_df["L_tilde_severe"].mean()),
        "L_tilde_p25": float(l.quantile(0.25)),
        "L_tilde_p75": float(l.quantile(0.75)),
        # Diagnostics (totals across all MC runs)
        "n_zero_loss_events": int(sim_df["n_zero_loss_events"].sum()),
        "max_disruption_radius_km": float(sim_df["max_disruption_radius_km"].max()),
        "n_fallback_events": int(sim_df["n_fallback_events"].sum()),
        # Legacy alias
        "node_normalized_loss": float(l.mean()),
    }
    return row
