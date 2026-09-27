#!/usr/bin/env python3
"""
Outage epicenters from census-tract population ONLY (network-independent).

Critical design
---------------
Epicenters must NOT be drawn near charging stations (no station-KDE).
P(hit) is then the genuine spatial overlap between the outage footprint and
the EV network. Sampling:

  1. Restrict to tracts in the outage county (FIPS).
  2. Draw a tract with probability ∝ 2020 Decennial population.
  3. Draw a uniform point inside that tract polygon.

Requires ``data/processed/pop_units_epicenter.gpkg`` (build with
``build_pop_units_epicenter.py``). Missing cache → hard error (do not fall back).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

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


def resolve_pop_units_path(project_root: Path) -> Path | None:
    candidates = [
        project_root / "data" / "processed" / "pop_units_epicenter.gpkg",
        project_root / "data" / "processed" / "pop_units_epicenter.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_pop_units_if_available(project_root: Path):
    """Load tract pop units; return None if missing (caller should error)."""
    path = resolve_pop_units_path(project_root)
    if path is None or gpd is None:
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    if "county_fips" not in gdf.columns:
        if "GEOID" in gdf.columns:
            gdf["county_fips"] = gdf["GEOID"].astype(str).str.zfill(11).str[:5]
        elif "tract_geoid" in gdf.columns:
            gdf["county_fips"] = gdf["tract_geoid"].astype(str).str.zfill(11).str[:5]
        else:
            raise ValueError(f"{path} missing county_fips / GEOID / tract_geoid")
    gdf["county_fips"] = gdf["county_fips"].astype(str).str.zfill(5)
    if "population" not in gdf.columns:
        for c in ("POPESTIMATE2023", "pop", "P1_001N"):
            if c in gdf.columns:
                gdf["population"] = gdf[c]
                break
    if "population" not in gdf.columns:
        raise ValueError(f"{path} missing population column")
    gdf["population"] = gdf["population"].astype(float).clip(lower=0.0)
    return gdf


def require_pop_units(project_root: Path):
    """Load tract units or raise with build instructions."""
    gdf = load_pop_units_if_available(project_root)
    if gdf is None or gdf.empty:
        raise FileNotFoundError(
            "Census-tract population units required for epicenters "
            "(network-independent). Build once:\n"
            "  python analysis/attack_under_PO/build_pop_units_epicenter.py\n"
            "Expected: data/processed/pop_units_epicenter.gpkg\n"
            "Station-KDE / station-anchored epicenters are disallowed "
            "(they inflate P(hit))."
        )
    return gdf


def choose_population_weighted_epicenter(
    county_fips: str,
    pop_gdf,
    schema: Optional[PopulationUnitSchema] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Sample tract ∝ population within county, then uniform point in tract."""
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
    total_w = float(weights.sum())
    probs = weights / total_w if total_w > 0 else None
    chosen_idx = int(rng.choice(np.arange(len(sub)), p=probs))
    poly = sub.iloc[chosen_idx][schema.geometry_col]
    return sample_point_in_polygon(poly, rng)


def choose_epicenter_population_mode(
    *,
    county_fips: str,
    poly,
    rng: np.random.Generator,
    G=None,  # unused — kept for call-site compatibility; MUST NOT influence draw
    pop_gdf=None,
    schema: Optional[PopulationUnitSchema] = None,
    sigma_km: float = 3.0,  # unused (legacy kw)
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Returns ((lat, lon) or None, method_tag).

    method_tag is always ``tract`` on success. Never uses the charging network.
    If ``pop_gdf`` is None → error. If county has no tracts → fall back to
    uniform in county polygon only as last resort (method=county_uniform),
    still network-independent.
    """
    del G, sigma_km  # explicitly ignore network / KDE
    fips = str(county_fips).zfill(5)

    if pop_gdf is None:
        raise RuntimeError(
            "pop_gdf is required for population epicenters. "
            "Call require_pop_units(project_root) before MC/CRN."
        )

    pt = choose_population_weighted_epicenter(fips, pop_gdf, schema=schema, rng=rng)
    if pt is not None:
        return (float(pt.y), float(pt.x)), "tract"

    # Rare: county missing from tract table — still no network
    if poly is not None and Point is not None:
        p = sample_point_in_polygon(poly, rng)
        return (float(p.y), float(p.x)), "county_uniform"

    return None, "none"


__all__ = [
    "PopulationUnitSchema",
    "load_pop_units_if_available",
    "require_pop_units",
    "choose_population_weighted_epicenter",
    "choose_epicenter_population_mode",
    "sample_point_in_polygon",
    "resolve_pop_units_path",
]
