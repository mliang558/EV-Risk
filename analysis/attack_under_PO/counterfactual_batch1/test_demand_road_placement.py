"""Focused tests for the population-targeted road placement rule."""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon

from analysis.attack_under_PO.counterfactual_batch1.densify_network import (
    _demand_road_candidates,
    _select_demand_sites,
)


def _inputs():
    tracts = gpd.GeoDataFrame(
        {
            "tract_geoid": ["00000000001", "00000000002", "00000000003"],
            "county_fips": ["00000"] * 3,
            "population": [3000, 2000, 1000],
        },
        geometry=[
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0)]),
            Polygon([(4, 0), (4, 1), (5, 1), (5, 0)]),
        ],
        crs=4326,
    )
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0.25), (1, 0.25)]),
            LineString([(2, 0.25), (3, 0.25)]),
        ],
        crs=4326,
    )
    return tracts, roads


def test_candidates_are_inside_tracts_and_on_roads():
    tracts, roads = _inputs()
    candidates = _demand_road_candidates(tracts, roads)
    assert candidates["tract_geoid"].tolist() == ["00000000001", "00000000002"]
    for candidate in candidates.itertuples(index=False):
        point = gpd.points_from_xy([candidate.lon], [candidate.lat])[0]
        tract = tracts.loc[tracts["tract_geoid"] == candidate.tract_geoid].geometry.iloc[0]
        assert tract.covers(point)
        assert roads.geometry.union_all().distance(point) < 1e-9


def test_selection_follows_population_order_and_adds_on_road_candidates():
    tracts, roads = _inputs()
    sites, eligible, candidates, _ = _select_demand_sites(
        tracts,
        roads,
        existing=np.asarray([[20.0, 20.0]]),
        target=3,
        min_distance_m=10_000,
    )
    assert len(candidates) > 2
    assert eligible >= 3
    assert len(sites) == 3
    assert sites[0, 1] < sites[1, 1]
