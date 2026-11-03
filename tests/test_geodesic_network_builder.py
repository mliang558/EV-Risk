from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from build_charging_network_step2 import build_region_graph, cluster_stations  # noqa: E402
from build_10km_panel_2018_2026 import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NETWORK_BUILD_DATE,
    YEARS_DEFAULT,
)


def _cluster_counts(coords: np.ndarray) -> list[int]:
    counts = cluster_stations(coords, np.ones(len(coords)), radius_m=10_000.0)[2]
    return sorted(counts.tolist())


def test_clustering_is_invariant_to_input_row_order() -> None:
    coords = np.array(
        [
            [40.0, -77.0],
            [40.0, -76.90],
            [40.0, -76.80],
            [40.3, -77.0],
        ]
    )
    expected = _cluster_counts(coords)
    shuffled = _cluster_counts(coords[[2, 0, 3, 1]])
    assert shuffled == expected


def test_ten_km_threshold_uses_geodesic_distance() -> None:
    # At 45 degrees north, these longitude offsets are approximately 9 km and
    # 11 km geodesically. EPSG:3857 would incorrectly magnify both distances.
    near = np.array([[45.0, -77.0], [45.0, -76.886]])
    far = np.array([[45.0, -77.0], [45.0, -76.860]])
    assert _cluster_counts(near) == [2]
    assert _cluster_counts(far) == [1, 1]


def test_combined_unit_is_built_as_one_connected_graph() -> None:
    df = pd.DataFrame(
        {
            "state": ["CT", "RI", "CT", "RI"],
            "lat": [41.5, 41.5, 41.8, 41.8],
            "lon": [-72.70, -72.64, -72.20, -72.14],
            "n_l1": [0, 0, 0, 0],
            "n_l2": [1, 1, 1, 1],
            "n_dc": [0, 0, 0, 0],
            "capacity": [25, 25, 25, 25],
        }
    )
    graph, meta = build_region_graph(
        df,
        ["CT", "RI"],
        "Connecticut",
        radius_m=10_000.0,
        min_nodes=1,
    )
    assert graph is not None and meta is not None
    assert meta["n_raw"] == 4
    assert meta["n_nodes"] == 2  # each cross-border pair forms one hypernode
    assert meta["member_states"] == ["CT", "RI"]
    assert meta["cluster_distance"] == "WGS84_geodesic"
    assert "+proj=aeqd" in meta["voronoi_crs"]
    assert nx.is_connected(graph)


def test_panel_defaults_identify_the_reviewed_six_year_network_data() -> None:
    assert YEARS_DEFAULT == [2018, 2019, 2020, 2021, 2022, 2023]
    assert NETWORK_BUILD_DATE == "2026-10-29"
    assert DEFAULT_OUT_ROOT.name.startswith("1029-network-data-preprocessing-")


if __name__ == "__main__":
    test_clustering_is_invariant_to_input_row_order()
    test_ten_km_threshold_uses_geodesic_distance()
    test_combined_unit_is_built_as_one_connected_graph()
    test_panel_defaults_identify_the_reviewed_six_year_network_data()
    print("PASS: geodesic clustering and combined-unit construction")

