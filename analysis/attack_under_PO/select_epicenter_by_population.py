#!/usr/bin/env python3
"""
Utilities for sampling outage epicenters using population (or a population proxy).

Within the outage county (already chosen by the λ-bootstrap), we sample an
epicenter so that denser / more populated places are more likely. Missing a
charging hypernode is *not* a bug — most real outages do not hit EV stations;
zero-loss events are substantive.

Methods (first available wins)
-----------------------------
1. ``tract`` — census-tract polygons with population
   (``data/processed/pop_units_epicenter.gpkg`` if present).
2. ``station_kde`` — Gaussian mixture centered on existing hypernodes in the
   county (σ default 3 km). Stations track population; this raises hit rates
   vs uniform county sampling while still allowing misses (recommended default
   when tracts are unavailable).
3. ``county_uniform`` — uniform in the county polygon (last resort).

Used by ``monte_carlo_outage_attack.py`` and CF Batch-1 CRN engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import networkx as nx

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover
    gpd = None
    Point = None  # type: ignore


@dataclass
class PopulationUnitSchema:
    county_fips_col: str = "county_fips"
    population_col: str = "population"
    geometry_col: str = "geometry"


def sample_point_in_polygon(poly, rng: np.random.Generator):
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(100):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            return p
    return poly.representative_point()


def load_population_units(path: str | Path):
    if gpd is None:
        raise ImportError("geopandas required for population units")
    gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        raise ValueError("Population GeoDataFrame must contain a 'geometry' column.")
    return gdf


def resolve_pop_units_path(project_root: Path) -> Path | None:
    candidates = [
        project_root / "data" / "processed" / "pop_units_epicenter.gpkg",
        project_root / "data" / "processed" / "pop_units_epicenter.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def choose_population_weighted_epicenter(
    county_fips: str,
    pop_gdf,
    schema: Optional[PopulationUnitSchema] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Tract / block units: sample unit ∝ population, then uniform in polygon."""
    if schema is None:
        schema = PopulationUnitSchema()
    if rng is None:
        rng = np.random.default_rng()

    fips_str = str(county_fips).zfill(5)
    sub = pop_gdf[
        pop_gdf[schema.county_fips_col].astype(str).str.zfill(5) == fips_str
    ].copy()
    if sub.empty:
        return None

    weights = np.asarray(sub[schema.population_col], dtype=float).clip(min=0.0)
    total_w = weights.sum()
    probs = weights / total_w if total_w > 0 else None
    chosen_idx = rng.choice(np.arange(len(sub)), p=probs)
    poly = sub.iloc[chosen_idx][schema.geometry_col]
    return sample_point_in_polygon(poly, rng)


def _stations_in_county(
    G: nx.Graph,
    county_fips: str,
    county_attr: str = "county_fips",
) -> list[Tuple[float, float]]:
    target = str(county_fips).zfill(5)
    out: list[Tuple[float, float]] = []
    for _, data in G.nodes(data=True):
        if data.get("is_cf_added") or data.get("coverage_expansion"):
            continue
        fips = data.get(county_attr)
        if fips is None or str(fips).zfill(5) != target:
            continue
        loc = data.get("location")
        if loc is not None and isinstance(loc, (tuple, list)) and len(loc) == 2:
            out.append((float(loc[0]), float(loc[1])))
        elif "lat" in data and "lon" in data:
            out.append((float(data["lat"]), float(data["lon"])))
    return out


def choose_station_kde_epicenter(
    G: nx.Graph,
    county_fips: str,
    poly,
    rng: np.random.Generator,
    *,
    sigma_km: float = 3.0,
    county_attr: str = "county_fips",
    max_tries: int = 80,
) -> Optional[Tuple[float, float]]:
    """
    Population proxy: pick a station in-county, jitter by N(0, σ) in km,
    reject if outside county polygon. Raises hit rate vs uniform while
    still allowing zero-loss events.
    """
    stations = _stations_in_county(G, county_fips, county_attr=county_attr)
    if not stations:
        return None
    # degrees ≈ km / 111
    sig_deg = float(sigma_km) / 111.0
    for _ in range(max_tries):
        lat0, lon0 = stations[int(rng.integers(0, len(stations)))]
        lat = float(lat0 + rng.normal(0.0, sig_deg))
        # longitude scale by cos(lat)
        lon = float(lon0 + rng.normal(0.0, sig_deg / max(0.2, np.cos(np.deg2rad(lat0)))))
        if poly is not None and Point is not None:
            if not poly.contains(Point(lon, lat)):
                continue
        return lat, lon
    # fallback: exact station (still a hit — rare)
    lat0, lon0 = stations[int(rng.integers(0, len(stations)))]
    return float(lat0), float(lon0)


def choose_epicenter_population_mode(
    *,
    county_fips: str,
    poly,
    rng: np.random.Generator,
    G: nx.Graph | None = None,
    pop_gdf=None,
    schema: Optional[PopulationUnitSchema] = None,
    sigma_km: float = 3.0,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Returns ((lat, lon) or None, method_tag).

    method_tag ∈ {tract, station_kde, county_uniform, none}
    """
    fips = str(county_fips).zfill(5)

    if pop_gdf is not None:
        pt = choose_population_weighted_epicenter(fips, pop_gdf, schema=schema, rng=rng)
        if pt is not None:
            # shapely Point: x=lon, y=lat
            return (float(pt.y), float(pt.x)), "tract"

    if G is not None:
        kde = choose_station_kde_epicenter(
            G, fips, poly, rng, sigma_km=sigma_km
        )
        if kde is not None:
            return kde, "station_kde"

    if poly is not None and Point is not None:
        p = sample_point_in_polygon(poly, rng)
        return (float(p.y), float(p.x)), "county_uniform"

    return None, "none"


def load_pop_units_if_available(project_root: Path):
    """Load tract pop units if cached; else None (station_kde will be used)."""
    path = resolve_pop_units_path(project_root)
    if path is None or gpd is None:
        return None
    gdf = gpd.read_file(path)
    # normalize columns
    if "county_fips" not in gdf.columns:
        if "GEOID" in gdf.columns:
            gdf["county_fips"] = gdf["GEOID"].astype(str).str.zfill(11).str[:5]
        elif "tract_geoid" in gdf.columns:
            gdf["county_fips"] = gdf["tract_geoid"].astype(str).str.zfill(11).str[:5]
    if "population" not in gdf.columns:
        for c in ("POPESTIMATE2023", "pop", "P1_001N"):
            if c in gdf.columns:
                gdf["population"] = gdf[c]
                break
    return gdf


__all__ = [
    "PopulationUnitSchema",
    "load_population_units",
    "load_pop_units_if_available",
    "choose_population_weighted_epicenter",
    "choose_station_kde_epicenter",
    "choose_epicenter_population_mode",
    "sample_point_in_polygon",
    "resolve_pop_units_path",
]
