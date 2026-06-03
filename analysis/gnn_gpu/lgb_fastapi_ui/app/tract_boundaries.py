"""Census tract polygon fills (urban / suburban / rural) — area layer, not stations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.map_geo import STRATUM_COLORS

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

try:
    from census_tract_strata import load_or_build_tract_strata, load_state_tracts, tract_cache_path, tract_shp_cache_dir
    from states import STATES_ORDER
except ImportError:
    load_or_build_tract_strata = None  # type: ignore
    load_state_tracts = None  # type: ignore
    tract_cache_path = None  # type: ignore
    tract_shp_cache_dir = None  # type: ignore
    STATES_ORDER = []

# Approx state bounds (south, west, north, east) for bbox → state filter
_STATE_BBOX: dict[str, tuple[float, float, float, float]] = {
    "AL": (30.0, -89.0, 35.0, -84.5),
    "AZ": (31.0, -115.0, 37.0, -109.0),
    "AR": (33.0, -94.5, 36.5, -89.5),
    "CA": (32.5, -125.0, 42.0, -114.0),
    "CO": (37.0, -109.5, 41.5, -102.0),
    "CT": (41.0, -73.8, 42.1, -71.8),
    "DE": (38.4, -75.8, 39.9, -75.0),
    "DC": (38.8, -77.1, 39.0, -76.9),
    "FL": (24.5, -87.5, 31.0, -80.0),
    "GA": (30.5, -85.5, 35.0, -80.8),
    "ID": (42.0, -117.5, 49.0, -111.0),
    "IL": (37.0, -91.5, 42.5, -87.5),
    "IN": (37.8, -88.5, 41.8, -84.8),
    "IA": (40.4, -96.6, 43.5, -90.1),
    "KS": (37.0, -102.5, 40.0, -94.6),
    "KY": (36.5, -89.6, 39.2, -82.0),
    "LA": (29.0, -94.0, 33.0, -89.0),
    "ME": (43.0, -71.1, 47.5, -66.9),
    "MD": (37.9, -79.5, 39.7, -75.0),
    "MA": (41.2, -73.5, 42.9, -69.9),
    "MI": (41.7, -90.4, 48.3, -82.4),
    "MN": (43.5, -97.2, 49.4, -89.5),
    "MS": (30.0, -91.7, 35.0, -88.1),
    "MO": (36.0, -95.8, 40.6, -89.1),
    "MT": (44.4, -116.1, 49.0, -104.0),
    "NE": (40.0, -104.1, 43.0, -95.3),
    "NV": (35.0, -120.0, 42.0, -114.0),
    "NH": (42.7, -72.6, 45.3, -70.6),
    "NJ": (38.9, -75.6, 41.4, -73.9),
    "NM": (31.3, -109.1, 37.0, -103.0),
    "NY": (40.5, -79.8, 45.0, -71.8),
    "NC": (33.8, -84.3, 36.6, -75.5),
    "ND": (45.9, -104.1, 49.0, -96.5),
    "OH": (38.4, -84.8, 42.0, -80.5),
    "OK": (33.6, -103.0, 37.0, -94.4),
    "OR": (42.0, -124.6, 46.3, -116.5),
    "PA": (39.7, -80.5, 42.3, -74.7),
    "RI": (41.1, -71.9, 42.0, -71.1),
    "SC": (32.0, -83.4, 35.2, -78.5),
    "SD": (42.5, -104.1, 45.9, -96.4),
    "TN": (35.0, -90.3, 36.7, -81.6),
    "TX": (25.8, -106.7, 36.5, -93.5),
    "UT": (37.0, -114.1, 42.0, -109.0),
    "VT": (42.7, -73.4, 45.0, -71.5),
    "VA": (36.5, -83.7, 39.5, -75.2),
    "WA": (45.5, -124.8, 49.0, -116.9),
    "WV": (37.2, -82.6, 40.6, -77.7),
    "WI": (42.5, -92.9, 47.1, -86.8),
    "WY": (41.0, -111.1, 45.0, -104.0),
}


def resolve_project_root(data_dir: Path) -> Path:
    for cand in (
        data_dir.parent.parent,
        data_dir.parent,
        Path(__file__).resolve().parents[3],
    ):
        if (cand / "data" / "processed").is_dir() or (cand / "analysis").is_dir():
            return cand
    return data_dir.parent.parent


def _analysis_on_path(project_root: Path) -> None:
    analysis = project_root / "analysis"
    if analysis.is_dir() and str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))


def _bbox_intersects(
    sb: float, wb: float, nb: float, eb: float,
    s2: float, w2: float, n2: float, e2: float,
) -> bool:
    return not (nb < s2 or sb > n2 or eb < w2 or wb > e2)


def states_in_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    south, west, north, east = bbox
    out = []
    for abbr, (s, w, n, e) in _STATE_BBOX.items():
        if _bbox_intersects(south, west, north, east, s, w, n, e):
            out.append(abbr)
    return out or list(STATES_ORDER)


def _load_strata_table(project_root: Path, data_dir: Path) -> pd.DataFrame | None:
    if tract_cache_path is None:
        return None
    path = tract_cache_path(project_root)
    if path.is_file():
        return load_or_build_tract_strata(project_root)
    alt = data_dir / "census_tract_strata.csv"
    if alt.is_file():
        return pd.read_csv(alt, dtype={"tract_geoid": str})
    nodes_p = data_dir / "windows_nodes.parquet"
    if nodes_p.is_file():
        nodes = pd.read_parquet(nodes_p)
        if "tract_geoid" in nodes.columns and "stratum" in nodes.columns:
            m = (
                nodes.dropna(subset=["tract_geoid"])
                .groupby("tract_geoid")["stratum"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "suburban")
                .reset_index()
            )
            m["tract_geoid"] = m["tract_geoid"].astype(str)
            return m.rename(columns={"stratum": "stratum"})
    return None


def tract_strata_geojson(
    data_dir: Path,
    bbox: tuple[float, float, float, float],
    *,
    max_features: int = 4000,
    simplify_tol: float = 0.008,
    allow_download: bool = True,
) -> dict[str, Any]:
    """GeoJSON polygons for census tracts colored by stratum (area fill layer)."""
    if load_state_tracts is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": "Missing lgb_fastapi_ui/census_tract_strata.py — re-upload UI folder",
        }

    import geopandas as gpd
    from shapely.geometry import box

    project_root = resolve_project_root(data_dir)
    _analysis_on_path(project_root)

    strata = _load_strata_table(project_root, data_dir)
    if strata is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": "Missing census_tract_strata.csv — run build_census_tract_strata.py",
        }

    stratum_col = "stratum" if "stratum" in strata.columns else "node_stratum"
    south, west, north, east = bbox
    clip_box = box(west, south, east, north)
    shp_dir = tract_shp_cache_dir(project_root)
    states = states_in_bbox(bbox)
    downloaded: list[str] = []
    parts: list[gpd.GeoDataFrame] = []

    for abbr in states:
        cache = shp_dir / f"{abbr}_tract2020.parquet"
        if not cache.is_file():
            if not allow_download:
                continue
            try:
                load_state_tracts(abbr, project_root)
                downloaded.append(abbr)
            except Exception:
                continue
        if not cache.is_file():
            continue
        try:
            tr = gpd.read_parquet(cache)
        except Exception:
            continue
        tr = tr.to_crs("EPSG:4326")
        try:
            sub = tr[tr.intersects(clip_box)]
        except Exception:
            sub = tr.cx[west:east, south:north]
        if len(sub):
            parts.append(sub)

    if not parts:
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": (
                "No tract polygons in cache. On server run: "
                "python lgb_fastapi_ui/download_tract_shapes.py "
                "(needs pygris + network, ~30–60 min once)"
            ),
            "states_in_view": states,
            "downloaded_now": downloaded,
        }

    gdf = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.merge(
        strata[["tract_geoid", stratum_col]].drop_duplicates("tract_geoid"),
        on="tract_geoid",
        how="left",
    )
    gdf["stratum"] = gdf[stratum_col].fillna("suburban").astype(str).str.lower()

    if simplify_tol > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)

    if len(gdf) > max_features:
        gdf = gdf.iloc[:max_features]

    features: list[dict[str, Any]] = []
    for row in gdf.itertuples(index=False):
        st = str(row.stratum).lower()
        if st not in STRATUM_COLORS:
            st = "suburban"
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        col = STRATUM_COLORS[st]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "tract_geoid": str(row.tract_geoid),
                    "stratum": st,
                    "color": col,
                    "fillOpacity": 0.5,
                },
                "geometry": geom.__geo_interface__,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "n_features": len(features),
        "bbox": [south, west, north, east],
        "states_loaded": states,
        "downloaded_now": downloaded,
    }
