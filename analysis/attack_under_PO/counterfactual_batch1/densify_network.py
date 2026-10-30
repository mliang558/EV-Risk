"""Add eligible Interstate-corridor hypernodes and rebuild the network."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys

import networkx as nx
import numpy as np

ANALYSIS = Path(__file__).resolve().parents[3] / "analysis"
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
CACHE_SCHEMA = "interstate_50mi_10km_unit_boundary_v3"
ROOT = Path(__file__).resolve().parents[3]
EV_PROJECT = Path(__file__).resolve().parents[4]


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


def resolve_interstate_shapefile() -> Path:
    candidates = (
        EV_PROJECT / "Highway" / "us_interstate_highways.shp" / "us_interstate_highways.shp",
        EV_PROJECT / "Highway" / "us_interstate_highways.shp",
        EV_PROJECT / "PrimaryRoads" / "tl_2022_us_primaryroads" / "tl_2022_us_primaryroads.shp",
        ROOT / "data" / "raw" / "us_interstate_highways.shp",
    )
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  ".join(map(str, candidates))
    raise FileNotFoundError(f"Interstate shapefile not found. Tried:\n  {tried}")


def _interstates_near(graph: nx.Graph, boundary=None):
    import geopandas as gpd
    from shapely.geometry import box

    if boundary is None:
        coords, _ = _node_coords(graph)
        lat_min, lon_min = coords.min(axis=0)
        lat_max, lon_max = coords.max(axis=0)
        boundary = box(lon_min - 0.35, lat_min - 0.35, lon_max + 0.35, lat_max + 0.35)

    road_path = resolve_interstate_shapefile()
    roads = gpd.read_file(road_path)
    roads = roads.set_crs(4326) if roads.crs is None else roads.to_crs(4326)
    if "RTTYP" in roads.columns:
        roads = roads[roads["RTTYP"].astype(str).str.upper() == "I"].copy()
    elif "FULLNAME" in roads.columns:
        roads = roads[roads["FULLNAME"].astype(str).str.contains(
            r"I-|Interstate", case=False, na=False
        )].copy()
    elif road_path.name.lower() != "us_interstate_highways.shp":
        raise ValueError(
            "Road file has neither RTTYP nor FULLNAME; Interstate segments "
            "cannot be identified safely"
        )
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
    line_parts = []
    for geometry in projected.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            line_parts.append(geometry)
        elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
            line_parts.extend(g for g in geometry.geoms if g.geom_type == "LineString")
    if not line_parts:
        return np.empty((0, 2))
    merged = linemerge(unary_union(line_parts))
    lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    points = []
    for line in lines:
        distances = np.arange(0, line.length + 1, spacing_m)
        points.extend(line.interpolate(min(d, line.length)) for d in distances)
    if not points:
        return np.empty((0, 2))
    points = gpd.GeoSeries(points, crs=projection).to_crs(4326)
    return np.column_stack((points.y, points.x))


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
    order = eligible[np.argsort(-initial_distance[eligible])]
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
    candidates = _corridor_candidates(_interstates_near(graph, boundary), spacing_m)
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
    return expanded, {
        "mode": "nevi",
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
    if path.is_file() and not force:
        with path.open("rb") as file:
            payload = pickle.load(file)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        compatible = (
            meta.get("cache_schema") == CACHE_SCHEMA
            and meta.get("base_graph_signature") == signature
            and meta.get("boundary_signature") == boundary_signature
            and meta.get("cluster_radius_m") == float(cluster_radius_m)
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
