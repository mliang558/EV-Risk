"""Build corridor- and population-targeted expansion networks."""

from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
import pickle
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

def _project_root() -> Path:
    marker = Path("analysis/attack_under_PO")
    for candidate in (Path.cwd(), Path(__file__).absolute().parents[3]):
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(f"Run from the project checkout containing {marker}")


ROOT = _project_root()
ANALYSIS = ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from analysis.build_charging_network_step2 import (
    WGS84_GEOD,
    _geodesic_distance_matrix,
    _to_local_aeqd_meters,
    voronoi_adjacency,
)

CLUSTER_RADIUS_M = 10_000.0
CORRIDOR_SPACING_M = 50 * 1609.344
CACHE_SCHEMA = "fhwa_ev_afc_50mi_10km_unit_boundary_v4"
DEMAND_CACHE_SCHEMA = "population_primary_road_matched_count_10km_v6"
DEMAND_CANDIDATE_SPACING_M = CLUSTER_RADIUS_M / 2
AFC_FILENAME = "ntad_ev_alternative_fuel_corridors_round8_2025.geojson"
PRIMARY_ROADS_FILENAME = "tl_2023_us_primaryroads.zip"
PRIMARY_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/"
    + PRIMARY_ROADS_FILENAME
)


def _node_coords(graph: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = list(graph.nodes)
    coords = []
    for node in nodes:
        data = graph.nodes[node]
        location = data.get("location")
        lat, lon = (
            location if location is not None else (data.get("lat"), data.get("lon"))
        )
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Node {node!r} has no valid latitude/longitude") from error
        if not np.isfinite([lat, lon]).all():
            raise ValueError(f"Node {node!r} has no valid latitude/longitude")
        coords.append((lat, lon))
    return np.asarray(coords), nodes


def _graph_signature(graph: nx.Graph) -> str:
    coords, _ = _node_coords(graph)
    coords = np.round(coords, 10)
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    digest = hashlib.sha256(coords.astype("<f8", copy=False).tobytes()).hexdigest()
    return f"n={len(coords)};sha256={digest}"


def _boundary_signature(boundary) -> str:
    if boundary is None:
        return "none"
    return hashlib.sha256(boundary.wkb).hexdigest()


@lru_cache(maxsize=8)
def _file_signature_text(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> str:
    return _file_signature_text(str(path.absolute()))


def _rebuild_edges(graph: nx.Graph) -> nx.Graph:
    """Use the same topology and distance definitions as the canonical network."""
    coords, nodes = _node_coords(graph)
    projected, projection = _to_local_aeqd_meters(coords)
    adjacency = (
        {(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))}
        if len(nodes) < 4
        else voronoi_adjacency(projected)
    )
    distances = _geodesic_distance_matrix(coords)
    max_distance = max(float(distances.max()), 1.0)

    rebuilt = nx.Graph(**graph.graph)
    rebuilt.add_nodes_from((node, dict(graph.nodes[node])) for node in nodes)
    for i, j in adjacency:
        distance = float(distances[i, j])
        rebuilt.add_edge(
            nodes[i], nodes[j], distance_m=distance, weight=distance / max_distance
        )
    rebuilt.graph.update(
        voronoi_crs=projection,
        adjacency="complete_small" if len(nodes) < 4 else "voronoi_local_aeqd",
        distance_method="WGS84_geodesic",
    )
    if nodes and not nx.is_connected(rebuilt):
        raise RuntimeError("Counterfactual network is disconnected")
    return rebuilt


def resolve_ev_afc_file() -> Path:
    """Return the pinned official USDOT/BTS EV Alternative Fuel Corridors file."""
    configured = os.environ.get("EV_AFC_PATH", "").strip()
    path = Path(configured) if configured else ROOT / "data" / "raw" / AFC_FILENAME
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Official EV AFC file not found: {path}. Run "
            "download_ev_afc.py before the intervention experiment."
        )
    return path


@lru_cache(maxsize=1)
def _load_ev_afc(path_text: str):
    import geopandas as gpd

    roads = gpd.read_file(path_text)
    roads = roads.set_crs(4326) if roads.crs is None else roads.to_crs(4326)
    columns = {str(column).upper(): column for column in roads.columns}
    required = {"EV", "ELECTRICVE", "STATE", "AFC_NUMBER"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Official EV AFC file is missing fields: {missing}")
    roads = roads[
        pd.to_numeric(roads[columns["EV"]], errors="coerce").fillna(0) > 0
    ].copy()
    if roads.empty:
        raise ValueError(f"No designated EV corridors found in {path_text}")
    invalid_geometry = ~roads.geometry.geom_type.isin(
        ["LineString", "MultiLineString"]
    )
    if invalid_geometry.any():
        raise ValueError("Official EV AFC file contains non-line geometry")
    return roads, columns


def validate_ev_afc_file() -> dict:
    path = resolve_ev_afc_file()
    roads, columns = _load_ev_afc(str(path.absolute()))
    return {
        "path": str(path.absolute()),
        "sha256": _file_signature(path),
        "n_segments": int(len(roads)),
        "statuses": sorted(roads[columns["ELECTRICVE"]].dropna().astype(str).unique()),
    }


def resolve_primary_roads_file() -> Path:
    """Return the pinned national Census TIGER/Line primary-roads archive."""
    configured = os.environ.get("TIGER_PRIMARY_ROADS_PATH", "").strip()
    path = Path(configured) if configured else ROOT / "data" / "raw" / PRIMARY_ROADS_FILENAME
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Census primary-roads file not found: {path}. Run "
            "download_primary_roads.py before the Demand experiment."
        )
    return path


def _vector_source(path: Path) -> str:
    return f"zip://{path.absolute().as_posix()}" if path.suffix.lower() == ".zip" else str(path)


def validate_primary_roads_file() -> dict:
    """Validate the official road source without loading the full national layer."""
    import geopandas as gpd

    path = resolve_primary_roads_file()
    roads = gpd.read_file(_vector_source(path), rows=20)
    if roads.empty:
        raise ValueError(f"No road segments found in {path}")
    if not roads.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("Census primary-roads file contains non-line geometry")
    return {
        "path": str(path.absolute()),
        "url": PRIMARY_ROADS_URL,
        "sha256": _file_signature(path),
        "vintage": 2023,
        "layer": "TIGER/Line Primary Roads",
    }


def load_primary_roads_near(boundary):
    """Read and clip national primary roads to one analytical unit."""
    import geopandas as gpd

    if boundary is None or boundary.is_empty:
        raise ValueError("Demand placement requires a non-empty analytical-unit boundary")
    source = validate_primary_roads_file()
    roads = gpd.read_file(_vector_source(Path(source["path"])), bbox=boundary.bounds)
    roads = roads.set_crs(4326) if roads.crs is None else roads.to_crs(4326)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads[roads.intersects(boundary)].copy()
    roads["geometry"] = roads.geometry.intersection(boundary)
    roads = roads[~roads.geometry.is_empty].copy()
    if roads.empty:
        raise ValueError("No Census primary-road segments intersect this analytical unit")
    return roads, source


def _ev_corridors_near(graph: nx.Graph, boundary=None):
    from shapely.geometry import box

    if boundary is None:
        coords, _ = _node_coords(graph)
        lat_min, lon_min = coords.min(axis=0)
        lat_max, lon_max = coords.max(axis=0)
        boundary = box(lon_min - 0.35, lat_min - 0.35, lon_max + 0.35, lat_max + 0.35)

    road_path = resolve_ev_afc_file()
    roads, columns = _load_ev_afc(str(road_path.absolute()))
    roads = roads.copy()
    state = roads[columns["STATE"]].fillna("").astype(str)
    corridor = roads[columns["AFC_NUMBER"]].fillna("").astype(str)
    roads["_corridor_id"] = state + ":" + corridor
    roads = roads[roads.intersects(boundary)].copy()
    roads["geometry"] = roads.geometry.intersection(boundary)
    return roads[~roads.geometry.is_empty]


def _corridor_candidates(roads, spacing_m: float) -> np.ndarray:
    """Sample route points in a local equidistant projection."""
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union

    if roads.empty:
        return np.empty((0, 2))
    roads = roads.to_crs(4326)
    union = (
        roads.geometry.union_all()
        if hasattr(roads.geometry, "union_all")
        else roads.geometry.unary_union
    )
    center = union.centroid
    projection = (
        f"+proj=aeqd +lat_0={center.y:.12f} +lon_0={center.x:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    projected = roads.to_crs(projection)
    points = []
    for _, group in projected.groupby("_corridor_id", sort=True):
        parts = []
        for geometry in group.geometry:
            if geometry is None or geometry.is_empty:
                continue
            if geometry.geom_type == "LineString":
                parts.append(geometry)
            elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
                parts.extend(g for g in geometry.geoms if g.geom_type == "LineString")
        if not parts:
            continue
        unioned = unary_union(parts)
        merged = unioned if unioned.geom_type == "LineString" else linemerge(unioned)
        lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
        for line in lines:
            if line.length <= spacing_m:
                distances = [line.length / 2]
            else:
                distances = np.arange(spacing_m / 2, line.length, spacing_m)
            points.extend(line.interpolate(float(distance)) for distance in distances)
    if not points:
        return np.empty((0, 2))
    points = gpd.GeoSeries(points, crs=projection).to_crs(4326)
    coordinates = np.column_stack((points.y, points.x))
    return np.unique(np.round(coordinates, 10), axis=0)


def _minimum_distances(candidates: np.ndarray, references: np.ndarray) -> np.ndarray:
    if not len(candidates):
        return np.empty(0)
    if not len(references):
        return np.full(len(candidates), np.inf)

    result = np.empty(len(candidates))
    for start in range(0, len(candidates), 500):
        chunk = candidates[start : start + 500]
        n_chunk, n_ref = len(chunk), len(references)
        _, _, distance = WGS84_GEOD.inv(
            np.repeat(chunk[:, 1], n_ref),
            np.repeat(chunk[:, 0], n_ref),
            np.tile(references[:, 1], n_chunk),
            np.tile(references[:, 0], n_chunk),
        )
        result[start : start + n_chunk] = np.asarray(distance).reshape(n_chunk, n_ref).min(1)
    return result


def _select_sites(
    candidates: np.ndarray,
    existing: np.ndarray,
    target: int,
    min_distance_m: float,
) -> tuple[np.ndarray, int]:
    """Take largest-gap candidates while enforcing spacing from every chosen site."""
    initial_distance = _minimum_distances(candidates, existing)
    eligible = np.flatnonzero(initial_distance >= min_distance_m)
    order = eligible[
        np.lexsort(
            (
                candidates[eligible, 1],
                candidates[eligible, 0],
                -initial_distance[eligible],
            )
        )
    ]
    selected = []
    occupied = existing.copy()
    for index in order:
        point = candidates[index : index + 1]
        if _minimum_distances(point, occupied)[0] < min_distance_m:
            continue
        selected.append(point[0])
        occupied = np.vstack((occupied, point))
        if len(selected) == target:
            break
    sites = np.asarray(selected) if selected else np.empty((0, 2))
    return sites, len(eligible)


def build_corridor_network(
    graph: nx.Graph,
    pct: float,
    *,
    boundary=None,
    cluster_radius_m: float = CLUSTER_RADIUS_M,
    spacing_m: float = CORRIDOR_SPACING_M,
) -> tuple[nx.Graph, dict]:
    existing, _ = _node_coords(graph)
    target = int(np.ceil(pct * len(existing))) if pct > 0 else 0
    candidates = _corridor_candidates(_ev_corridors_near(graph, boundary), spacing_m)
    sites, eligible_count = _select_sites(
        candidates, existing, target, cluster_radius_m
    )

    expanded = graph.copy()
    for index, (lat, lon) in enumerate(sites):
        expanded.add_node(
            f"cf_nevi_{index}",
            lat=float(lat),
            lon=float(lon),
            location=(float(lat), float(lon)),
            n_stations=1,
            is_cf_added=True,
        )
    expanded = _rebuild_edges(expanded) if len(sites) else expanded
    added = len(sites)
    min_existing = (
        float(_minimum_distances(sites, existing).min()) if added else None
    )
    min_added = None
    if added > 1:
        _, _, distances = WGS84_GEOD.inv(
            np.repeat(sites[:, 1], added),
            np.repeat(sites[:, 0], added),
            np.tile(sites[:, 1], added),
            np.tile(sites[:, 0], added),
        )
        distances = np.asarray(distances).reshape(added, added)
        np.fill_diagonal(distances, np.inf)
        min_added = float(distances.min())
    return expanded, {
        "mode": "fhwa_ev_afc",
        "pct": float(pct),
        "n_base": len(existing),
        "n_target": target,
        "n_added": added,
        "n_nevi": added,
        "n_shortfall": target - added,
        "realized_pct": added / len(existing) if len(existing) else 0.0,
        "n_candidates": len(candidates),
        "n_candidates_eligible": eligible_count,
        "spacing_m": float(spacing_m),
        "cluster_radius_m": float(cluster_radius_m),
        "min_distance_to_existing_m": min_existing,
        "min_distance_between_added_m": min_added,
        "corridor_source": str(resolve_ev_afc_file().absolute()),
        "corridor_source_sha256": _file_signature(resolve_ev_afc_file()),
    }


def _cache_path(cache_root: Path, pct: float, unit: str) -> Path:
    dose = int(round(100 * pct))
    return cache_root / f"nevi_p{dose}" / "network_structures" / f"network_{unit}.pkl"


def load_or_build_densified(
    graph: nx.Graph,
    unit: str,
    pct: float,
    cache_root: Path,
    *,
    boundary=None,
    cluster_radius_m: float = CLUSTER_RADIUS_M,
    force: bool = False,
) -> tuple[nx.Graph, dict]:
    """Load a compatible corridor network or rebuild and cache it."""
    path = _cache_path(cache_root, pct, unit)
    signature = _graph_signature(graph)
    boundary_signature = _boundary_signature(boundary)
    corridor_signature = _file_signature(resolve_ev_afc_file())
    if path.is_file() and not force:
        with path.open("rb") as file:
            payload = pickle.load(file)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        compatible = (
            meta.get("cache_schema") == CACHE_SCHEMA
            and meta.get("base_graph_signature") == signature
            and meta.get("boundary_signature") == boundary_signature
            and meta.get("cluster_radius_m") == float(cluster_radius_m)
            and meta.get("corridor_source_sha256") == corridor_signature
        )
        if compatible:
            return payload["network"], meta
        print(f"[cache-rebuild] incompatible cache: {path}", flush=True)

    expanded, meta = build_corridor_network(
        graph, pct, boundary=boundary, cluster_radius_m=cluster_radius_m
    )
    meta.update(
        cache_schema=CACHE_SCHEMA,
        base_graph_signature=signature,
        boundary_signature=boundary_signature,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump({"network": expanded, "meta": meta}, file)
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return expanded, meta


def _population_signature(pop_units) -> str:
    """Hash population values and complete tract geometry."""
    digest = hashlib.sha256()
    sort_columns = [column for column in ("county_fips", "tract_geoid") if column in pop_units]
    ordered = pop_units.sort_values(sort_columns) if sort_columns else pop_units
    for index, row in ordered.iterrows():
        tract = str(row.get("tract_geoid", index))
        county = str(row.get("county_fips", "")).zfill(5)
        digest.update(f"{county}|{tract}|{float(row['population']):.6f}|".encode())
        digest.update(row.geometry.wkb)
    return digest.hexdigest()


def _demand_road_candidates(pop_units, primary_roads) -> pd.DataFrame:
    """Place one candidate per populated tract on a primary road inside it."""
    from shapely.ops import nearest_points, unary_union

    columns = ["population", "tract_geoid", "lat", "lon", "road_snap_distance_m"]
    if pop_units.empty or primary_roads.empty:
        return pd.DataFrame(columns=columns)

    tracts = pop_units.to_crs(4326).copy()
    roads = primary_roads.to_crs(4326).copy()
    road_index = roads.sindex

    records = []
    candidate_points = []
    for index, row in tracts.iterrows():
        population = float(row["population"])
        tract = row.geometry
        if population <= 0 or tract is None or tract.is_empty:
            continue
        matches = list(road_index.query(tract, predicate="intersects"))
        if not matches:
            continue
        pieces = [
            geometry.intersection(tract)
            for geometry in roads.geometry.iloc[matches]
            if geometry is not None and not geometry.is_empty
        ]
        pieces = [geometry for geometry in pieces if not geometry.is_empty and geometry.length > 0]
        if not pieces:
            continue
        roads_inside = unary_union(pieces)
        origin = tract.representative_point()
        road_point = nearest_points(origin, roads_inside)[1]
        candidate_points.append(road_point)
        _, _, snap_distance = WGS84_GEOD.inv(
            origin.x, origin.y, road_point.x, road_point.y
        )
        records.append(
            {
                "population": population,
                "tract_geoid": str(row.get("tract_geoid", index)),
                "road_snap_distance_m": float(snap_distance),
            }
        )

    if not records:
        return pd.DataFrame(columns=columns)
    for record, point in zip(records, candidate_points):
        record.update(lat=float(point.y), lon=float(point.x))
    return pd.DataFrame(records).sort_values(
        ["population", "tract_geoid", "lat", "lon"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _supplemental_demand_road_candidates(
    pop_units,
    primary_roads,
    spacing_m: float = DEMAND_CANDIDATE_SPACING_M,
) -> pd.DataFrame:
    """Sample extra on-road positions when one point per tract is insufficient."""
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union

    columns = ["population", "tract_geoid", "lat", "lon", "road_snap_distance_m"]
    if pop_units.empty or primary_roads.empty:
        return pd.DataFrame(columns=columns)
    if spacing_m <= 0:
        raise ValueError("Demand candidate spacing must be positive")

    tracts = pop_units.to_crs(4326).copy()
    roads = primary_roads.to_crs(4326).copy()
    min_lon, min_lat, max_lon, max_lat = tracts.total_bounds
    projection = (
        f"+proj=aeqd +lat_0={(min_lat + max_lat) / 2:.12f} "
        f"+lon_0={(min_lon + max_lon) / 2:.12f} +datum=WGS84 +units=m +no_defs"
    )
    tracts_m = tracts.to_crs(projection)
    roads_m = roads.to_crs(projection)
    road_index = roads_m.sindex

    records = []
    points = []
    for index, row in tracts_m.iterrows():
        population = float(row["population"])
        tract = row.geometry
        if population <= 0 or tract is None or tract.is_empty:
            continue
        matches = list(road_index.query(tract, predicate="intersects"))
        pieces = [
            geometry.intersection(tract)
            for geometry in roads_m.geometry.iloc[matches]
            if geometry is not None and not geometry.is_empty
        ]
        line_parts = []
        for geometry in pieces:
            if geometry.is_empty:
                continue
            if geometry.geom_type == "LineString":
                line_parts.append(geometry)
            elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
                line_parts.extend(
                    part for part in geometry.geoms if part.geom_type == "LineString"
                )
        if not line_parts:
            continue
        merged = unary_union(line_parts)
        if merged.geom_type not in ("LineString", "MultiLineString"):
            merged = linemerge(merged)
        lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
        for line in lines:
            if line.is_empty or line.length <= 0:
                continue
            distances = (
                [line.length / 2]
                if line.length <= spacing_m
                else np.arange(spacing_m / 2, line.length, spacing_m)
            )
            for distance in distances:
                points.append(line.interpolate(float(distance)))
                records.append(
                    {
                        "population": population,
                        "tract_geoid": str(row.get("tract_geoid", index)),
                        "road_snap_distance_m": 0.0,
                    }
                )

    if not records:
        return pd.DataFrame(columns=columns)
    points_wgs84 = gpd.GeoSeries(points, crs=projection).to_crs(4326)
    for record, point in zip(records, points_wgs84):
        record.update(lat=float(point.y), lon=float(point.x))
    candidates = pd.DataFrame(records)
    candidates["_lat_key"] = candidates["lat"].round(8)
    candidates["_lon_key"] = candidates["lon"].round(8)
    candidates = candidates.sort_values(
        ["population", "tract_geoid", "lat", "lon"],
        ascending=[False, True, True, True],
    ).drop_duplicates(["_lat_key", "_lon_key"])
    return candidates[columns].reset_index(drop=True)


def _greedy_candidate_indices(
    candidates: pd.DataFrame,
    existing: np.ndarray,
    order: np.ndarray,
    target: int,
    min_distance_m: float,
) -> np.ndarray:
    """Select a separated subset following one deterministic candidate order."""
    coordinates = candidates[["lat", "lon"]].to_numpy(dtype=float)
    available = _minimum_distances(coordinates, existing) >= min_distance_m
    selected = []
    for index in np.asarray(order, dtype=int):
        if not available[index]:
            continue
        selected.append(int(index))
        if len(selected) == target:
            break
        point = coordinates[index : index + 1]
        available &= _minimum_distances(coordinates, point) >= min_distance_m
    return np.asarray(selected, dtype=int)


def _matched_count_candidate_indices(
    candidates: pd.DataFrame,
    existing: np.ndarray,
    target: int,
    min_distance_m: float,
) -> np.ndarray:
    """Find a matched-count separated set, preferring higher-population sites."""
    if target <= 0 or candidates.empty:
        return np.empty(0, dtype=int)

    coordinates = candidates[["lat", "lon"]].to_numpy(dtype=float)
    population = candidates["population"].to_numpy(dtype=float)
    initial_distance = _minimum_distances(coordinates, existing)
    n_candidates = len(candidates)
    index = np.arange(n_candidates)
    orders = [
        np.lexsort((coordinates[:, 1], coordinates[:, 0], -population)),
        np.lexsort((coordinates[:, 1], coordinates[:, 0], -initial_distance)),
        np.lexsort((coordinates[:, 1], coordinates[:, 0])),
        np.lexsort((-coordinates[:, 1], -coordinates[:, 0])),
    ]

    # Deterministic multistart avoids a dense high-population cluster blocking
    # the matched node count. Weighted starts still favor populous tracts.
    rng = np.random.default_rng(20231102)
    population_weight = np.maximum(np.log1p(population), 1e-12)
    for _ in range(32):
        orders.append(rng.permutation(index))
        orders.append(np.argsort(rng.exponential(1 / population_weight)))

    best = np.empty(0, dtype=int)
    best_population = -np.inf
    for order in orders:
        selected = _greedy_candidate_indices(
            candidates, existing, order, target, min_distance_m
        )
        score = float(np.log1p(population[selected]).sum())
        if len(selected) > len(best) or (
            len(selected) == len(best) and score > best_population
        ):
            best, best_population = selected, score
        if len(best) == target and order is orders[0]:
            break
    if len(best) == target:
        return best
    exact = _exact_matched_count_candidate_indices(
        candidates, existing, target, min_distance_m
    )
    return exact if len(exact) == target else best


def _exact_matched_count_candidate_indices(
    candidates: pd.DataFrame,
    existing: np.ndarray,
    target: int,
    min_distance_m: float,
) -> np.ndarray:
    """Maximize covered population subject to exact count and 10 km conflicts."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, vstack

    coordinates = candidates[["lat", "lon"]].to_numpy(dtype=float)
    eligible = np.flatnonzero(
        _minimum_distances(coordinates, existing) >= min_distance_m
    )
    if len(eligible) < target:
        return np.empty(0, dtype=int)

    eligible_coordinates = coordinates[eligible]
    distances = _geodesic_distance_matrix(eligible_coordinates)
    edge_i, edge_j = np.where(np.triu(distances < min_distance_m, k=1))
    n_edges, n_sites = len(edge_i), len(eligible)
    rows = np.repeat(np.arange(n_edges), 2)
    columns = np.column_stack((edge_i, edge_j)).ravel()
    conflicts = coo_matrix(
        (np.ones(2 * n_edges), (rows, columns)), shape=(n_edges, n_sites)
    ).tocsr()
    count = csr_matrix(np.ones((1, n_sites)))
    exact_constraints = vstack((conflicts, count), format="csr")
    exact_lower = np.concatenate((np.full(n_edges, -np.inf), [float(target)]))
    exact_upper = np.concatenate((np.ones(n_edges), [float(target)]))
    population = candidates.iloc[eligible]["population"].to_numpy(dtype=float)

    result = milp(
        c=-np.log1p(population),
        integrality=np.ones(n_sites),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(exact_constraints, exact_lower, exact_upper),
        options={"time_limit": 180, "mip_rel_gap": 0.0},
    )
    selected = np.flatnonzero(result.x > 0.5) if result.x is not None else np.empty(0, int)
    if len(selected) == target:
        return eligible[selected]
    if result.status != 2:
        raise RuntimeError(
            "Exact Demand placement did not prove feasibility or infeasibility: "
            f"status={result.status}, message={result.message}"
        )

    # The requested count is infeasible. Solve the maximum-cardinality problem
    # so the reported capacity is exact rather than an arbitrary greedy shortfall.
    conflict_constraint = LinearConstraint(
        conflicts, np.full(n_edges, -np.inf), np.ones(n_edges)
    )
    capacity_result = milp(
        c=-np.ones(n_sites),
        integrality=np.ones(n_sites),
        bounds=Bounds(0, 1),
        constraints=conflict_constraint,
        options={"time_limit": 180, "mip_rel_gap": 0.0},
    )
    if capacity_result.status != 0 or capacity_result.x is None:
        raise RuntimeError(
            "Could not determine exact Demand placement capacity: "
            f"status={capacity_result.status}, message={capacity_result.message}"
        )
    capacity = int(np.rint(capacity_result.x).sum())

    # Among maximum-cardinality layouts, retain the one covering most population.
    capacity_constraints = vstack((conflicts, count), format="csr")
    capacity_lower = np.concatenate((np.full(n_edges, -np.inf), [float(capacity)]))
    capacity_upper = np.concatenate((np.ones(n_edges), [float(capacity)]))
    population_result = milp(
        c=-np.log1p(population),
        integrality=np.ones(n_sites),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            capacity_constraints, capacity_lower, capacity_upper
        ),
        options={"time_limit": 180, "mip_rel_gap": 0.0},
    )
    if population_result.x is None:
        selected = np.flatnonzero(capacity_result.x > 0.5)
    else:
        selected = np.flatnonzero(population_result.x > 0.5)
    print(
        f"[Demand capacity] requested={target} exact_maximum={capacity}",
        flush=True,
    )
    return eligible[selected]


def _select_demand_sites(
    pop_units,
    primary_roads,
    existing: np.ndarray,
    target: int,
    min_distance_m: float,
) -> tuple[np.ndarray, int, pd.DataFrame, np.ndarray]:
    """Select population-ranked road candidates subject to node spacing."""
    if target <= 0 or pop_units.empty:
        return np.empty((0, 2)), 0, pd.DataFrame(), np.empty(0)
    primary = _demand_road_candidates(pop_units, primary_roads)

    selected = []
    selected_snap_distances = []
    occupied = existing.copy()

    def add_from(table: pd.DataFrame) -> None:
        nonlocal occupied
        for row in table.itertuples(index=False):
            lat, lon = float(row.lat), float(row.lon)
            point = np.asarray([[lat, lon]], dtype=float)
            if _minimum_distances(point, occupied)[0] < min_distance_m:
                continue
            selected.append(point[0])
            selected_snap_distances.append(float(row.road_snap_distance_m))
            occupied = np.vstack((occupied, point))
            if len(selected) == target:
                return

    # Preserve the original rule whenever one on-road point per tract suffices.
    add_from(primary)
    supplemental = pd.DataFrame(columns=primary.columns)
    if len(selected) < target:
        supplemental = _supplemental_demand_road_candidates(
            pop_units, primary_roads, spacing_m=min_distance_m / 2
        )
        if not supplemental.empty and not primary.empty:
            primary_keys = set(zip(primary["lat"].round(8), primary["lon"].round(8)))
            supplemental = supplemental[
                [
                    (round(row.lat, 8), round(row.lon, 8)) not in primary_keys
                    for row in supplemental.itertuples(index=False)
                ]
            ].reset_index(drop=True)
        add_from(supplemental)

    candidates = pd.concat([primary, supplemental], ignore_index=True)
    if len(selected) < target:
        matched = _matched_count_candidate_indices(
            candidates, existing, target, min_distance_m
        )
        selected = candidates.loc[matched, ["lat", "lon"]].to_numpy(dtype=float).tolist()
        selected_snap_distances = (
            candidates.loc[matched, "road_snap_distance_m"].to_numpy(dtype=float).tolist()
        )
    if candidates.empty:
        eligible = 0
    else:
        coordinates = candidates[["lat", "lon"]].to_numpy(dtype=float)
        eligible = int((_minimum_distances(coordinates, existing) >= min_distance_m).sum())
    sites = np.asarray(selected) if selected else np.empty((0, 2))
    return sites, eligible, candidates, np.asarray(selected_snap_distances)


def build_demand_network(
    graph: nx.Graph,
    target: int,
    pop_units,
    primary_roads,
    road_source: dict,
    *,
    cluster_radius_m: float = CLUSTER_RADIUS_M,
) -> tuple[nx.Graph, dict]:
    """Add a matched number of nodes on roads in the most populated tracts."""
    existing, _ = _node_coords(graph)
    sites, eligible_count, candidates, snap_distances = _select_demand_sites(
        pop_units, primary_roads, existing, int(target), cluster_radius_m
    )
    if len(sites) != int(target):
        raise ValueError(
            "Demand comparator cannot match the corridor expansion: "
            f"target={int(target)}, selected={len(sites)}, "
            f"road_candidates={len(candidates)}, spacing_eligible={eligible_count}"
        )
    expanded = graph.copy()
    for index, (lat, lon) in enumerate(sites):
        expanded.add_node(
            f"cf_demand_{index}",
            lat=float(lat),
            lon=float(lon),
            location=(float(lat), float(lon)),
            n_stations=1,
            is_cf_added=True,
            placement_mode="population_primary_road",
        )
    expanded = _rebuild_edges(expanded) if len(sites) else expanded
    added = len(sites)
    min_added = None
    if added > 1:
        _, _, distances = WGS84_GEOD.inv(
            np.repeat(sites[:, 1], added),
            np.repeat(sites[:, 0], added),
            np.tile(sites[:, 1], added),
            np.tile(sites[:, 0], added),
        )
        distances = np.asarray(distances).reshape(added, added)
        np.fill_diagonal(distances, np.inf)
        min_added = float(distances.min())
    return expanded, {
        "mode": "population_primary_road_matched",
        "n_base": len(existing),
        "n_target": int(target),
        "n_added": added,
        "n_shortfall": int(target) - added,
        "realized_pct": added / len(existing) if len(existing) else 0.0,
        "n_population_tracts": int(len(pop_units)),
        "n_candidates": int(len(candidates)),
        "n_candidates_eligible": int(eligible_count),
        "candidate_spacing_m": float(cluster_radius_m / 2),
        "cluster_radius_m": float(cluster_radius_m),
        "min_distance_to_existing_m": (
            float(_minimum_distances(sites, existing).min()) if added else None
        ),
        "min_distance_between_added_m": min_added,
        "road_snap_distance_median": (
            float(np.median(snap_distances)) if len(snap_distances) else None
        ),
        "road_snap_distance_max_m": (
            float(np.max(snap_distances)) if len(snap_distances) else None
        ),
        "road_source": road_source,
        "population_source_signature": _population_signature(pop_units),
    }


def load_or_build_demand(
    graph: nx.Graph,
    unit: str,
    pct: float,
    target: int,
    pop_units,
    primary_roads,
    road_source: dict,
    cache_root: Path,
    *,
    cluster_radius_m: float = CLUSTER_RADIUS_M,
    force: bool = False,
) -> tuple[nx.Graph, dict]:
    """Load or build the population-targeted matched-count comparator."""
    dose = int(round(100 * pct))
    path = (
        cache_root
        / f"demand_p{dose}"
        / "network_structures"
        / f"network_{unit}.pkl"
    )
    signature = _graph_signature(graph)
    pop_signature = _population_signature(pop_units)
    road_signature = str(road_source["sha256"])
    if path.is_file() and not force:
        with path.open("rb") as file:
            payload = pickle.load(file)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        compatible = (
            meta.get("cache_schema") == DEMAND_CACHE_SCHEMA
            and meta.get("base_graph_signature") == signature
            and meta.get("population_source_signature") == pop_signature
            and meta.get("road_source", {}).get("sha256") == road_signature
            and int(meta.get("n_target", -1)) == int(target)
            and meta.get("cluster_radius_m") == float(cluster_radius_m)
        )
        if compatible:
            return payload["network"], meta
        print(f"[cache-rebuild] incompatible cache: {path}", flush=True)

    expanded, meta = build_demand_network(
        graph,
        target,
        pop_units,
        primary_roads,
        road_source,
        cluster_radius_m=cluster_radius_m,
    )
    meta.update(cache_schema=DEMAND_CACHE_SCHEMA, base_graph_signature=signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump({"network": expanded, "meta": meta}, file)
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return expanded, meta
