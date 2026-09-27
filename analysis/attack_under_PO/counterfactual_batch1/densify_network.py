#!/usr/bin/env python3
"""
CF-D coverage expansion: add hypernodes OUTSIDE existing 10 km cluster footprints.

Placement rules (all enforce min distance ≥ cluster_radius_m to existing hypernodes)
------------------------------------------------------------------------------------
nevi     : PRIMARY — candidates every 50 miles along interstate / AFC corridors;
           exclude sites < 10 km from an existing hypernode; at each dose take the
           largest-gap sites first (policy-relevant NEVI-style fill-in).
pop      : population-weighted rejection sampling in unit counties (demand-oriented).
uniform  : uniform in unit counties (null / spatial control); use seed replicates.

Capacity is irrelevant: global efficiency is topology-only. Within-cluster densify /
EVSE upgrades are beyond model resolution (see densify_within).

Base graphs are NOT rebuilt — densify from existing 10 km pickles
(``outputs/network_graph_10km_2018_2026/2023/...``).

Cached pickles:
  outputs/network_graph_cf_densify_2023/{nevi,pop,uniform}_p{pct}[_r{k}]/network_structures/
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi

try:
    from run_charging_network import _to_epsg3857_meters
except Exception:  # pragma: no cover
    def _to_epsg3857_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = lon * 20037508.34 / 180.0
        y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * 20037508.34 / np.pi
        return np.column_stack([x, y])


DEFAULT_CLUSTER_RADIUS_M = 10_000.0
NEVI_SPACING_M = 50.0 * 1609.344  # 50 miles — NEVI corridor spacing
NULL_REPLICATES = 10

# Sibling of Pro_directory (EV_Project) — highway + county assets live here
_EV_PROJECT = Path(__file__).resolve().parents[4]
_PRO_ROOT = Path(__file__).resolve().parents[3]


def _node_coords(G: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = list(G.nodes())
    coords = np.zeros((len(nodes), 2), dtype=float)
    for i, n in enumerate(nodes):
        d = G.nodes[n]
        if "location" in d and d["location"] is not None:
            lat, lon = d["location"]
        else:
            lat, lon = d.get("lat"), d.get("lon")
        coords[i] = (float(lat), float(lon))
    return coords, nodes


def _voronoi_or_delaunay_edges(coords_3857: np.ndarray) -> set[tuple[int, int]]:
    n = len(coords_3857)
    if n < 2:
        return set()
    if n < 4:
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    try:
        vor = Voronoi(coords_3857)
        adj: set[tuple[int, int]] = set()
        for p1, p2 in vor.ridge_points:
            a, b = (p1, p2) if p1 < p2 else (p2, p1)
            adj.add((a, b))
        return adj
    except Exception:
        tri = Delaunay(coords_3857)
        adj = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted((int(simplex[i]), int(simplex[(i + 1) % 3])))
                if a != b:
                    adj.add((a, b))
        return adj


def _rebuild_edges(G: nx.Graph) -> nx.Graph:
    coords_ll, nodes = _node_coords(G)
    coords_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])
    adj = _voronoi_or_delaunay_edges(coords_m)
    dist = np.linalg.norm(coords_m[:, None, :] - coords_m[None, :, :], axis=2)
    dmax = float(dist.max()) if dist.size else 1.0
    if dmax <= 0:
        dmax = 1.0
    H = nx.Graph()
    for n in nodes:
        H.add_node(n, **dict(G.nodes[n]))
    for a, b in adj:
        na, nb = nodes[a], nodes[b]
        d_m = float(dist[a, b])
        H.add_edge(na, nb, weight=d_m / dmax, distance_m=d_m)
    return H


def _add_cf_nodes(
    G: nx.Graph,
    lats: list[float],
    lons: list[float],
    *,
    prefix: str,
    rule: str,
) -> nx.Graph:
    """Attach new hypernodes (capacity placeholder only). Tag is_cf_added=True."""
    H = G.copy()
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        nid = f"{prefix}_{i}"
        H.add_node(
            nid,
            lat=float(lat),
            lon=float(lon),
            location=(float(lat), float(lon)),
            capacity=1.0,
            n_stations=1,
            n_l1=0.0,
            n_l2=0.0,
            n_dc=0.0,
            densify_mode=rule,
            coverage_expansion=True,
            is_cf_added=True,  # never eligible as epicenter
        )
    return _rebuild_edges(H)


def _n_target(n0: int, pct: float) -> int:
    if pct <= 0:
        return 0
    return max(1, int(np.ceil(pct * n0)))


# ---------------------------------------------------------------------------
# GIS helpers (lazy imports — geopandas / shapely)
# ---------------------------------------------------------------------------

def resolve_interstate_shapefile() -> Path:
    candidates = [
        _EV_PROJECT / "Highway" / "us_interstate_highways.shp" / "us_interstate_highways.shp",
        _EV_PROJECT / "Highway" / "us_interstate_highways.shp",
        _EV_PROJECT / "PrimaryRoads" / "tl_2022_us_primaryroads" / "tl_2022_us_primaryroads.shp",
        _PRO_ROOT / "data" / "raw" / "us_interstate_highways.shp",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Interstate / AFC corridor shapefile not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_county_pop_csv() -> Path:
    candidates = [
        _PRO_ROOT
        / "analysis"
        / "attack_under_PO"
        / "module_3_03"
        / "demographic data"
        / "merged_output.csv",
        _EV_PROJECT / "processed_county_population.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("County population CSV not found.")


def resolve_county_shapefile() -> Path:
    from compute_impact_radius import resolve_county_shapefile as _res

    return _res(_PRO_ROOT)


def _bbox_from_graph(G: nx.Graph, pad_deg: float = 0.35) -> tuple[float, float, float, float]:
    coords, _ = _node_coords(G)
    lat_min, lon_min = coords.min(axis=0)
    lat_max, lon_max = coords.max(axis=0)
    return (
        float(lon_min - pad_deg),
        float(lat_min - pad_deg),
        float(lon_max + pad_deg),
        float(lat_max + pad_deg),
    )


def _load_interstate_clipped(bbox: tuple[float, float, float, float]):
    import geopandas as gpd
    from shapely.geometry import box

    path = resolve_interstate_shapefile()
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    # Prefer true interstates when TIGER primary roads are used
    if "RTTYP" in gdf.columns:
        inter = gdf[gdf["RTTYP"].astype(str).str.upper() == "I"]
        if len(inter) > 0:
            gdf = inter
    elif "MTFCC" in gdf.columns and "FULLNAME" in gdf.columns:
        # keep Interstate-named S1100 segments
        mask = gdf["FULLNAME"].astype(str).str.contains("I-|Interstate", case=False, na=False)
        if mask.any():
            gdf = gdf[mask]
    clip = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
    try:
        out = gpd.clip(gdf, clip)
    except Exception:
        out = gdf[gdf.intersects(clip.geometry.iloc[0])]
    return out


def _load_unit_counties(G: nx.Graph, member_state_abbrs: list[str] | None):
    """Counties overlapping the unit bbox, optionally filtered by state abbr."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    # STATEFP (2-digit) → postal abbr (TIGER counties have STATEFP, not STUSPS)
    _FIPS_TO_ABBR = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
        "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
        "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
        "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
        "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
        "54": "WV", "55": "WI", "56": "WY",
    }

    shp = resolve_county_shapefile()
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(4269)
    gdf = gdf.to_crs(4326)
    if "GEOID" in gdf.columns:
        gdf["fips_str"] = gdf["GEOID"].astype(str).str.zfill(5)
    else:
        gdf["fips_str"] = (
            gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)
        )
    bbox = _bbox_from_graph(G)
    clip = box(*bbox)
    gdf = gdf[gdf.intersects(clip)].copy()
    if member_state_abbrs:
        abbrs = {a.upper() for a in member_state_abbrs}
        if "STUSPS" in gdf.columns:
            gdf = gdf[gdf["STUSPS"].astype(str).str.upper().isin(abbrs)]
        elif "STATEFP" in gdf.columns:
            keep_fips = {k for k, v in _FIPS_TO_ABBR.items() if v in abbrs}
            gdf = gdf[gdf["STATEFP"].astype(str).str.zfill(2).isin(keep_fips)]
    # attach population
    pop_path = resolve_county_pop_csv()
    pop = pd.read_csv(pop_path, dtype={"fips": str})
    if "fips" not in pop.columns and "fips_str" in pop.columns:
        pop = pop.rename(columns={"fips_str": "fips"})
    pop["fips"] = pop["fips"].astype(str).str.zfill(5)
    pop_col = None
    for c in ("POPESTIMATE2023", "POPESTIMATE2022", "POPESTIMATE2024", "population", "POP"):
        if c in pop.columns:
            pop_col = c
            break
    if pop_col is None:
        raise ValueError(f"No population column in {pop_path}")
    pop = pop[["fips", pop_col]].rename(columns={pop_col: "population"})
    gdf = gdf.merge(pop, left_on="fips_str", right_on="fips", how="left")
    gdf["population"] = gdf["population"].fillna(0).clip(lower=0)
    return gdf


def _sample_along_lines(
    roads_gdf,
    spacing_m: float = NEVI_SPACING_M,
) -> np.ndarray:
    """Return (N, 2) lon/lat points every spacing_m along each line."""
    if roads_gdf is None or len(roads_gdf) == 0:
        return np.zeros((0, 2), dtype=float)
    roads_m = roads_gdf.to_crs(3857)
    pts: list[tuple[float, float]] = []
    for geom in roads_m.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for line in lines:
            length = float(line.length)
            if length <= 0:
                continue
            n = max(1, int(np.floor(length / spacing_m)))
            for i in range(n + 1):
                d = min(i * spacing_m, length)
                p = line.interpolate(d)
                pts.append((p.x, p.y))
    if not pts:
        return np.zeros((0, 2), dtype=float)
    arr_m = np.asarray(pts, dtype=float)
    # 3857 → lon/lat
    import geopandas as gpd
    from shapely.geometry import Point

    g = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in arr_m], crs=3857).to_crs(4326)
    return np.column_stack([g.geometry.y, g.geometry.x])  # lat, lon


def _min_dist_m(cand_m: np.ndarray, existing_m: np.ndarray) -> np.ndarray:
    """Pairwise min distance from each cand to existing (meters, EPSG:3857)."""
    if len(cand_m) == 0:
        return np.zeros(0, dtype=float)
    if len(existing_m) == 0:
        return np.full(len(cand_m), np.inf)
    # chunk to limit memory
    out = np.empty(len(cand_m), dtype=float)
    chunk = 2000
    for i in range(0, len(cand_m), chunk):
        sl = cand_m[i : i + chunk]
        d = np.linalg.norm(sl[:, None, :] - existing_m[None, :, :], axis=2)
        out[i : i + chunk] = d.min(axis=1)
    return out


def _greedy_select_by_gap(
    cand_ll: np.ndarray,
    existing_m: np.ndarray,
    n_take: int,
    *,
    min_dist_m: float,
    mutual_min_m: float | None = None,
) -> tuple[list[float], list[float], np.ndarray]:
    """
    Keep candidates with min_dist ≥ min_dist_m; take up to n_take largest gaps.
    Optionally enforce mutual spacing among selected (default = min_dist_m).
    """
    if n_take <= 0 or len(cand_ll) == 0:
        return [], [], np.zeros(0)
    mutual = min_dist_m if mutual_min_m is None else mutual_min_m
    cand_m = _to_epsg3857_meters(cand_ll[:, 0], cand_ll[:, 1])
    dmin = _min_dist_m(cand_m, existing_m)
    ok = np.where(dmin >= min_dist_m)[0]
    if len(ok) == 0:
        return [], [], dmin
    order = ok[np.argsort(-dmin[ok])]  # largest gap first
    chosen_idx: list[int] = []
    chosen_m = existing_m.copy()
    for idx in order:
        if len(chosen_idx) >= n_take:
            break
        p = cand_m[idx]
        if len(chosen_m):
            if float(np.linalg.norm(chosen_m - p, axis=1).min()) < mutual:
                continue
        chosen_idx.append(int(idx))
        chosen_m = np.vstack([chosen_m, p])
    lats = [float(cand_ll[i, 0]) for i in chosen_idx]
    lons = [float(cand_ll[i, 1]) for i in chosen_idx]
    return lats, lons, dmin


# ---------------------------------------------------------------------------
# Placement rules
# ---------------------------------------------------------------------------

def densify_nevi(
    G: nx.Graph,
    pct: float,
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    spacing_m: float = NEVI_SPACING_M,
    member_state_abbrs: list[str] | None = None,
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """
    NEVI corridor (main): every ``spacing_m`` along interstate, drop sites
    within ``cluster_radius_m`` of existing hypernodes, take largest gaps.

    Dose target = ceil(pct * |V|). If corridor candidates run out (common in
    small states at 50%), remaining slots are filled with population-weighted
    placement (same rule as CF-D-pop). Meta records n_nevi / n_pop_fill.
    """
    coords_ll, nodes = _node_coords(G)
    n0 = len(nodes)
    n_add = _n_target(n0, pct)
    if n_add == 0:
        return G.copy(), {"n_added": 0, "mode": "nevi", "pct": pct, "n_target": 0}

    existing_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])
    bbox = _bbox_from_graph(G)
    roads = _load_interstate_clipped(bbox)
    cand_ll = _sample_along_lines(roads, spacing_m=spacing_m)
    lats, lons, dmin = _greedy_select_by_gap(
        cand_ll, existing_m, n_add, min_dist_m=cluster_radius_m
    )
    n_nevi = len(lats)
    n_short = n_add - n_nevi
    n_pop_fill = 0
    fill_note = ""

    H = _add_cf_nodes(G, lats, lons, prefix="cf_nevi", rule="nevi")

    if n_short > 0:
        # Fill remainder with population-weighted sites on the partially densified graph
        H_fill, meta_fill = densify_pop(
            H,
            pct=0.0,
            cluster_radius_m=cluster_radius_m,
            member_state_abbrs=member_state_abbrs,
            seed=int(seed) + 17,
            n_add_exact=n_short,
        )
        # densify_pop rebuilds from H; count new nodes
        n_pop_fill = int(meta_fill.get("n_added", 0))
        H = H_fill
        fill_note = (
            f" Corridor shortfall {n_short}; filled {n_pop_fill} via population-weighted rule."
        )

    meta = {
        "mode": "nevi",
        "intervention": "coverage_expansion_nevi",
        "pct": float(pct),
        "n_base": n0,
        "n_target": n_add,
        "n_added": int(n_nevi + n_pop_fill),
        "n_nevi": int(n_nevi),
        "n_pop_fill": int(n_pop_fill),
        "n_shortfall_before_fill": int(max(0, n_short)),
        "n_candidates": int(len(cand_ll)),
        "n_candidates_eligible": int((dmin >= cluster_radius_m).sum()) if len(dmin) else 0,
        "spacing_m": float(spacing_m),
        "cluster_radius_m": float(cluster_radius_m),
        "interstate_shp": str(resolve_interstate_shapefile()),
        "n_nodes": H.number_of_nodes(),
        "n_edges": H.number_of_edges(),
        "capacity_used_by_efficiency": False,
        "seed": int(seed),
        "note": (
            "NEVI-style: points every 50 mi on interstate corridors; "
            "exclude <10 km from existing hypernodes; dose = ceil(pct*|V|). "
            "If corridor candidates are exhausted, remaining slots use "
            "population-weighted placement."
            + fill_note
        ),
    }
    return H, meta


def densify_pop(
    G: nx.Graph,
    pct: float,
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    member_state_abbrs: list[str] | None = None,
    seed: int = 42,
    max_tries_factor: int = 200,
    n_add_exact: int | None = None,
) -> tuple[nx.Graph, dict]:
    """Population-weighted sampling in unit counties, >10 km from existing."""
    from shapely.geometry import Point

    coords_ll, nodes = _node_coords(G)
    n0 = len(nodes)
    n_add = int(n_add_exact) if n_add_exact is not None else _n_target(n0, pct)
    if n_add <= 0:
        return G.copy(), {"n_added": 0, "mode": "pop", "pct": pct}

    rng = np.random.default_rng(seed)
    counties = _load_unit_counties(G, member_state_abbrs)
    weights = counties["population"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(counties), dtype=float)
    probs = weights / weights.sum()
    existing_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])

    lats: list[float] = []
    lons: list[float] = []
    tries = 0
    max_tries = max(n_add * max_tries_factor, n_add * 50)
    while len(lats) < n_add and tries < max_tries:
        tries += 1
        i = int(rng.choice(len(counties), p=probs))
        poly = counties.geometry.iloc[i]
        if poly is None or poly.is_empty:
            continue
        # rejection sample inside polygon
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(40):
            lon = float(rng.uniform(minx, maxx))
            lat = float(rng.uniform(miny, maxy))
            if not poly.contains(Point(lon, lat)):
                continue
            cand_m = _to_epsg3857_meters(np.array([lat]), np.array([lon]))[0]
            dmin = float(np.linalg.norm(existing_m - cand_m, axis=1).min())
            if dmin < cluster_radius_m:
                break
            # mutual vs already chosen
            if lats:
                chosen_m = _to_epsg3857_meters(np.asarray(lats), np.asarray(lons))
                if float(np.linalg.norm(chosen_m - cand_m, axis=1).min()) < cluster_radius_m:
                    break
            lats.append(lat)
            lons.append(lon)
            existing_m = np.vstack([existing_m, cand_m])
            break

    H = _add_cf_nodes(G, lats, lons, prefix="cf_pop", rule="pop")
    meta = {
        "mode": "pop",
        "intervention": "coverage_expansion_pop",
        "pct": float(pct),
        "n_base": n0,
        "n_added": len(lats),
        "n_target": n_add,
        "n_counties": int(len(counties)),
        "cluster_radius_m": float(cluster_radius_m),
        "seed": int(seed),
        "tries": tries,
        "n_nodes": H.number_of_nodes(),
        "n_edges": H.number_of_edges(),
        "capacity_used_by_efficiency": False,
        "note": "Population-weighted sites in unit counties, >10 km from existing hypernodes.",
    }
    return H, meta


def densify_uniform(
    G: nx.Graph,
    pct: float,
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    member_state_abbrs: list[str] | None = None,
    seed: int = 42,
    max_tries_factor: int = 200,
) -> tuple[nx.Graph, dict]:
    """
    Uniform spatial null inside unit counties (not statewide bbox — still
    restricted to county polygons). Forests/deserts can be selected by design;
    use as control only (typically 10 seed replicates).
    """
    from shapely.geometry import Point

    coords_ll, nodes = _node_coords(G)
    n0 = len(nodes)
    n_add = _n_target(n0, pct)
    if n_add == 0:
        return G.copy(), {"n_added": 0, "mode": "uniform", "pct": pct}

    rng = np.random.default_rng(seed)
    counties = _load_unit_counties(G, member_state_abbrs)
    # area-weighted for "uniform in space"
    areas = counties.geometry.to_crs(3857).area.to_numpy(dtype=float)
    areas = np.clip(areas, 0, None)
    if areas.sum() <= 0:
        areas = np.ones(len(counties), dtype=float)
    probs = areas / areas.sum()
    existing_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])

    lats: list[float] = []
    lons: list[float] = []
    tries = 0
    max_tries = max(n_add * max_tries_factor, n_add * 50)
    while len(lats) < n_add and tries < max_tries:
        tries += 1
        i = int(rng.choice(len(counties), p=probs))
        poly = counties.geometry.iloc[i]
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(40):
            lon = float(rng.uniform(minx, maxx))
            lat = float(rng.uniform(miny, maxy))
            if not poly.contains(Point(lon, lat)):
                continue
            cand_m = _to_epsg3857_meters(np.array([lat]), np.array([lon]))[0]
            dmin = float(np.linalg.norm(existing_m - cand_m, axis=1).min())
            if dmin < cluster_radius_m:
                break
            if lats:
                chosen_m = _to_epsg3857_meters(np.asarray(lats), np.asarray(lons))
                if float(np.linalg.norm(chosen_m - cand_m, axis=1).min()) < cluster_radius_m:
                    break
            lats.append(lat)
            lons.append(lon)
            existing_m = np.vstack([existing_m, cand_m])
            break

    H = _add_cf_nodes(G, lats, lons, prefix="cf_null", rule="uniform")
    meta = {
        "mode": "uniform",
        "intervention": "coverage_expansion_null",
        "pct": float(pct),
        "n_base": n0,
        "n_added": len(lats),
        "n_target": n_add,
        "cluster_radius_m": float(cluster_radius_m),
        "seed": int(seed),
        "tries": tries,
        "n_nodes": H.number_of_nodes(),
        "n_edges": H.number_of_edges(),
        "capacity_used_by_efficiency": False,
        "note": "Uniform null control inside county polygons; may land in low-demand areas.",
    }
    return H, meta


def densify_within(
    G: nx.Graph,
    pct: float,
    *,
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """Diagnostic only — topology unchanged (beyond 10 km resolution)."""
    if pct <= 0:
        return G.copy(), {"n_boosted": 0, "mode": "within", "pct": pct}

    rng = np.random.default_rng(seed)
    H = G.copy()
    nodes = list(H.nodes())
    n0 = len(nodes)
    n_boost = max(1, int(np.ceil(pct * n0)))
    chosen = rng.choice(nodes, size=min(n_boost, n0), replace=False)
    for n in chosen:
        H.nodes[n]["densify_mode"] = "within_diagnostic"
        H.nodes[n]["n_stations"] = int(H.nodes[n].get("n_stations", 1) or 1) + 1
    meta = {
        "mode": "within",
        "intervention": "beyond_model_resolution",
        "pct": float(pct),
        "n_base": n0,
        "n_boosted": int(len(chosen)),
        "n_nodes": H.number_of_nodes(),
        "n_edges": H.number_of_edges(),
        "seed": int(seed),
        "note": (
            "Within-cluster densification / capacity is beyond 10 km hypernode resolution; "
            "topology unchanged so efficiency-based L is unchanged."
        ),
    }
    return H, meta


# Back-compat alias (old name "gap" → NEVI)
def densify_gap(G: nx.Graph, pct: float, **kwargs: Any) -> tuple[nx.Graph, dict]:
    return densify_nevi(G, pct, **kwargs)


def densify_graph(
    G: nx.Graph,
    pct: float,
    mode: str,
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    seed: int = 42,
    member_state_abbrs: list[str] | None = None,
) -> tuple[nx.Graph, dict]:
    mode = mode.lower().strip()
    # aliases
    if mode in ("gap", "nevi", "corridor", "afc"):
        return densify_nevi(
            G,
            pct,
            cluster_radius_m=cluster_radius_m,
            member_state_abbrs=member_state_abbrs,
            seed=seed,
        )
    if mode in ("pop", "population"):
        return densify_pop(
            G,
            pct,
            cluster_radius_m=cluster_radius_m,
            member_state_abbrs=member_state_abbrs,
            seed=seed,
        )
    if mode in ("uniform", "null", "random"):
        return densify_uniform(
            G,
            pct,
            cluster_radius_m=cluster_radius_m,
            member_state_abbrs=member_state_abbrs,
            seed=seed,
        )
    if mode == "within":
        return densify_within(G, pct, seed=seed)
    raise ValueError(f"Unknown densify mode: {mode!r}")


def cache_path(
    cache_root: Path,
    mode: str,
    pct: float,
    unit: str,
    *,
    replicate: int | None = None,
) -> Path:
    tag = f"{mode}_p{int(round(pct * 100))}"
    if replicate is not None and mode in ("uniform", "null", "random"):
        tag = f"{tag}_r{int(replicate)}"
    return cache_root / tag / "network_structures" / f"network_{unit}.pkl"


def load_or_build_densified(
    G: nx.Graph,
    unit: str,
    pct: float,
    mode: str,
    cache_root: Path,
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    seed: int = 42,
    replicate: int | None = None,
    member_state_abbrs: list[str] | None = None,
    force: bool = False,
) -> tuple[nx.Graph, dict]:
    mode_norm = mode.lower().strip()
    if mode_norm in ("gap", "corridor", "afc"):
        mode_norm = "nevi"
    if mode_norm in ("population",):
        mode_norm = "pop"
    if mode_norm in ("null", "random"):
        mode_norm = "uniform"
    path = cache_path(cache_root, mode_norm, pct, unit, replicate=replicate)
    if path.is_file() and not force:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta", {}) or {}
        meta_path = path.with_suffix(".meta.json")
        if meta and not meta_path.is_file():
            try:
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass
        return payload["network"], meta

    # For null replicates, fold replicate into seed
    use_seed = int(seed)
    if replicate is not None:
        use_seed = int(seed) + 1000 * int(replicate)

    H, meta = densify_graph(
        G,
        pct,
        mode_norm,
        cluster_radius_m=cluster_radius_m,
        seed=use_seed,
        member_state_abbrs=member_state_abbrs,
    )
    if replicate is not None:
        meta["replicate"] = int(replicate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"network": H, "meta": meta}, f)
    # Sidecar for smoke tests (n_nevi + n_pop_fill == n_target)
    meta_path = path.with_suffix(".meta.json")
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass
    return H, meta
