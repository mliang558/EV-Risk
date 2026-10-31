"""Add eligible Interstate-corridor hypernodes and rebuild the network."""

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
AFC_FILENAME = "ntad_ev_alternative_fuel_corridors_round8_2025.geojson"


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


def _file_signature(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
