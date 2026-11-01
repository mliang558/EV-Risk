from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
ATTACK = ROOT / "analysis" / "attack_under_PO"
PKG = ATTACK / "counterfactual_batch1"
for path in (ATTACK, PKG.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from counterfactual_batch1.crn_engine import aggregate_sim_metrics
from attack_with_bootstrapped_outages import global_efficiency_fixed_denominator
from sample_lambda_from_posterior import (
    aggregate_state_lambda,
    bootstrap_state_outages,
    draw_state_event_counts,
)


def test_state_rate_is_sum_of_county_rates() -> None:
    counties = pd.DataFrame(
        {
            "state": ["A", "A", "B"],
            "lambda_draw": [2.0, 3.5, 7.0],
        }
    )
    state = aggregate_state_lambda(counties).set_index("state")
    assert state.loc["A", "lambda_state_sum_draw"] == 5.5
    assert state.loc["A", "n_counties"] == 2
    assert state.loc["B", "lambda_state_sum_draw"] == 7.0


def test_state_event_draw_is_reproducible() -> None:
    state = pd.DataFrame({"state": ["A"], "lambda_state_sum_draw": [20.0]})
    first = draw_state_event_counts(state, random_state=123)
    second = draw_state_event_counts(state, random_state=123)
    assert first["n_events_draw"].tolist() == second["n_events_draw"].tolist()


def test_counties_are_selected_in_proportion_to_lambda() -> None:
    state = pd.DataFrame({"state": ["A"], "n_events_draw": [10_000]})
    counties = pd.DataFrame(
        {
            "state": ["A", "A"],
            "fips": [1001, 1003],
            "lambda_draw": [1.0, 3.0],
        }
    )
    outages = pd.DataFrame(
        {
            "state": ["A", "A"],
            "fips": [1001, 1003],
            "duration_hours": [1.0, 1.0],
            "mean_customers": [10.0, 20.0],
        }
    )
    sampled = bootstrap_state_outages(state, counties, outages, random_state=7)
    share_second = (sampled["sim_county_fips"] == "01003").mean()
    assert abs(share_second - 0.75) < 0.02
    assert not sampled["used_state_pool_fallback"].any()


def test_loss_metric_identities() -> None:
    events = pd.DataFrame(
        {
            "sim_id": [0, 0, 0],
            "scenario": ["baseline"] * 3,
            "family": ["baseline"] * 3,
            "n_nodes": [10] * 3,
            "baseline_efficiency": [0.4] * 3,
            "rel_loss_x_duration": [1.0, 0.0, 2.0],
            "rel_lcc_loss_x_duration": [0.5, 0.0, 1.0],
            "hit": [1, 0, 1],
            "pct_eff_loss": [10.0, 0.0, 20.0],
            "lcc_frac": [0.9, 1.0, 0.8],
            "n_disrupted_nodes": [1, 0, 2],
            "used_state_pool_fallback": [False, False, False],
            "radius_km": [2.0, 1.0, 3.0],
        }
    )
    result = aggregate_sim_metrics(events)
    assert np.isclose(result["total_rel_loss_x_duration"], 3.0)
    assert np.isclose(result["L_tilde"], 0.3)
    assert np.isclose(result["P_hit"], 2.0 / 3.0)
    assert np.isclose(result["L_event"], 1.0)
    assert np.isclose(result["E_loss_given_hit"], 1.5)


def test_attacked_efficiency_uses_original_node_denominator() -> None:
    graph = nx.path_graph(4)
    attacked = graph.copy()
    attacked.remove_node(0)
    fixed = global_efficiency_fixed_denominator(attacked, graph.number_of_nodes())
    manual = nx.global_efficiency(attacked) * (3 * 2) / (4 * 3)
    assert np.isclose(fixed, manual)
    assert fixed < nx.global_efficiency(graph)
