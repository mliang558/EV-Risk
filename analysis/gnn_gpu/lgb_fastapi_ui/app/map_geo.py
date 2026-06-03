"""Map layers: clustered stations + sliding-window disks (no center markers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

STRATUM_RADIUS_KM: dict[str, float] = {
    "urban": 10.0,
    "suburban": 30.0,
    "rural": 80.0,
}

STRATUM_COLORS: dict[str, str] = {
    "urban": "#22c55e",
    "suburban": "#eab308",
    "rural": "#a855f7",
}


def haversine_km_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = np.pi / 180.0
    lat1r, lon1r = lat1 * r, lon1 * r
    lat2r, lon2r = lat2 * r, lon2 * r
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a))))


def haversine_km_array(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = np.pi / 180.0
    lat1r, lat2r = lat1 * r, lat2 * r
    dlat = lat2r - lat1r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def classify_location(
    lat: float,
    lon: float,
    *,
    stations: pd.DataFrame,
    meta: pd.DataFrame | None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Pick urban / suburban / rural and radius (10 / 30 / 80 km).

    Same source as charging-node labels (when Step-3 ran with census tract):
      1. FCC geocode click → tract GEOID → census_tract_strata.csv  (identical to nodes)
      2. Nearest station's node_stratum from windows_nodes.parquet
      3. Station-density heuristic (fallback only)
    """
    lat, lon = float(lat), float(lon)

    try:
        from census_tract_strata import stratum_at_lat_lon

        tract_hit = stratum_at_lat_lon(lat, lon, project_root)
        if tract_hit.get("window_stratum"):
            return tract_hit  # type: ignore[return-value]
    except Exception:
        pass

    if stations is not None and len(stations) and "node_stratum" in stations.columns:
        sub = stations[stations["node_stratum"].astype(str).str.len() > 0]
        if len(sub):
            d = haversine_km_array(lat, lon, sub["lat"].values.astype(float), sub["lon"].values.astype(float))
            j = int(np.argmin(d))
            if float(d[j]) <= 50.0:
                st = str(sub.iloc[j]["node_stratum"]).lower()
                if st in STRATUM_RADIUS_KM:
                    out = {
                        "lat": lat,
                        "lon": lon,
                        "window_stratum": st,
                        "radius_km": STRATUM_RADIUS_KM[st],
                        "method": "nearest_station_node_stratum",
                        "nearest_station_km": float(d[j]),
                        "nearest_node_id": str(sub.iloc[j].get("node_id", "")),
                    }
                    if "tract_geoid" in sub.columns:
                        out["tract_geoid"] = str(sub.iloc[j].get("tract_geoid", "") or "")
                    return out

    if stations is not None and len(stations):
        n10 = int((haversine_km_array(lat, lon, stations["lat"].values, stations["lon"].values) <= 10.0).sum())
        n30 = int((haversine_km_array(lat, lon, stations["lat"].values, stations["lon"].values) <= 30.0).sum())
        if n10 >= 15:
            st = "urban"
        elif n30 >= 15:
            st = "suburban"
        else:
            st = "rural"
        return {
            "lat": lat,
            "lon": lon,
            "window_stratum": st,
            "radius_km": STRATUM_RADIUS_KM[st],
            "method": "density_heuristic_fallback",
            "stations_within_10km": n10,
            "stations_within_30km": n30,
            "note": "Re-run Step-3 with census_tract_strata for tract-based labels on nodes.",
        }

    if meta is not None and len(meta) and {"center_lat", "center_lon"}.issubset(meta.columns):
        m = meta.copy()
        d = np.array(
            [
                haversine_km_scalar(lat, lon, float(r.center_lat), float(r.center_lon))
                for r in m.itertuples(index=False)
            ]
        )
        j = int(np.argmin(d))
        row = m.iloc[j]
        st = str(row.get("window_stratum", "suburban"))
        return {
            "lat": lat,
            "lon": lon,
            "window_stratum": st,
            "radius_km": float(STRATUM_RADIUS_KM.get(st, 30.0)),
            "method": "nearest_window_label",
            "nearest_window_id": str(row["window_id"]),
            "nearest_window_km": float(d[j]),
            "note": "window_stratum is a sampling tier (10/30/80 km), not census tract geography.",
        }

    return {
        "lat": lat,
        "lon": lon,
        "window_stratum": "suburban",
        "radius_km": 30.0,
        "method": "default",
    }


def stratum_legend_payload() -> dict[str, Any]:
    return {
        "strata": [
            {
                "id": k,
                "label": f"{k.capitalize()} ({int(v)} km)",
                "radius_km": v,
                "color": STRATUM_COLORS[k],
            }
            for k, v in STRATUM_RADIUS_KM.items()
        ]
    }


def _bbox_filter(df: pd.DataFrame, bbox: tuple[float, float, float, float] | None) -> pd.DataFrame:
    if bbox is None or df.empty:
        return df
    south, west, north, east = bbox
    lat = df["lat"].astype(float)
    lon = df["lon"].astype(float)
    return df[(lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)]


def load_stations_table(data_dir: Path) -> pd.DataFrame:
    """Unique charging nodes (post network cluster) from Step-3 windows_nodes."""
    nodes_path = data_dir / "windows_nodes.parquet"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"Missing {nodes_path}")
    nodes = pd.read_parquet(nodes_path)
    cols = ["node_id", "lat", "lon"]
    for c in ("state", "capacity", "node_stratum", "stratum", "tract_geoid"):
        if c in nodes.columns:
            cols.append(c)
    stations = nodes.drop_duplicates(subset=["node_id"]).loc[:, cols].copy()
    if "node_stratum" not in stations.columns and "stratum" in stations.columns:
        stations["node_stratum"] = stations["stratum"].astype(str)
    stations["lat"] = stations["lat"].astype(float)
    stations["lon"] = stations["lon"].astype(float)
    return stations


def load_windows_meta(data_dir: Path) -> pd.DataFrame:
    meta_path = data_dir / "windows_meta.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = pd.read_csv(meta_path)
    need = {"window_id", "center_lat", "center_lon", "radius_km"}
    if not need.issubset(meta.columns):
        raise ValueError(f"windows_meta missing columns: {need - set(meta.columns)}")
    return meta


def stations_payload(
    stations: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int = 20000,
    stratum: str | None = None,
) -> dict[str, Any]:
    df = stations
    if stratum and "node_stratum" in df.columns:
        df = df[df["node_stratum"].astype(str) == stratum]
    df = _bbox_filter(df, bbox)
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    points = []
    for r in df.itertuples(index=False):
        ns = str(getattr(r, "node_stratum", "") or getattr(r, "stratum", "") or "")
        points.append(
            {
                "node_id": str(r.node_id),
                "lat": float(r.lat),
                "lon": float(r.lon),
                "state": str(getattr(r, "state", "")),
                "capacity": float(getattr(r, "capacity", 0.0)),
                "node_stratum": ns,
                "stratum": ns.lower() if ns else "",
                "color": STRATUM_COLORS.get(ns.lower(), "#6b7280") if ns else "#6b7280",
            }
        )
    return {"n_points": len(points), "points": points}


def windows_payload(
    meta: pd.DataFrame,
    pred_by_id: dict[str, dict[str, float]] | None = None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int = 5000,
    window_stratum: str | None = None,
    radius_km: float | None = None,
) -> dict[str, Any]:
    df = meta.copy()
    if window_stratum and "window_stratum" in df.columns:
        df = df[df["window_stratum"].astype(str) == window_stratum]
    if radius_km is not None and "radius_km" in df.columns:
        df = df[np.isclose(df["radius_km"].astype(float), float(radius_km), atol=0.5)]
    if bbox is not None:
        tmp = df.rename(columns={"center_lat": "lat", "center_lon": "lon"})
        tmp = _bbox_filter(tmp, bbox)
        df = df.loc[tmp.index]
    if len(df) > limit:
        df = df.head(limit)
    windows = []
    for r in df.itertuples(index=False):
        wid = str(r.window_id)
        extra = pred_by_id.get(wid, {}) if pred_by_id else {}
        windows.append(
            {
                "window_id": wid,
                "center_lat": float(r.center_lat),
                "center_lon": float(r.center_lon),
                "radius_km": float(r.radius_km),
                "radius_m": float(r.radius_km) * 1000.0,
                "window_stratum": str(getattr(r, "window_stratum", "")),
                "n_nodes": int(getattr(r, "n_nodes", 0)),
                **extra,
            }
        )
    return {"n_windows": len(windows), "windows": windows}


def window_detail(
    data_dir: Path,
    window_id: str,
    meta: pd.DataFrame,
    nodes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    row = meta[meta["window_id"].astype(str) == str(window_id)]
    if row.empty:
        raise KeyError(window_id)
    r = row.iloc[0]
    out: dict[str, Any] = {
        "window_id": str(window_id),
        "center_lat": float(r["center_lat"]),
        "center_lon": float(r["center_lon"]),
        "radius_km": float(r["radius_km"]),
        "radius_m": float(r["radius_km"]) * 1000.0,
        "window_stratum": str(r.get("window_stratum", "")),
        "n_nodes": int(r.get("n_nodes", 0)),
    }
    if nodes is not None:
        sub = nodes[nodes["window_id"].astype(str) == str(window_id)]
        out["stations"] = [
            {
                "node_id": str(x.node_id),
                "lat": float(x.lat),
                "lon": float(x.lon),
                "state": str(getattr(x, "state", "")),
                "capacity": float(getattr(x, "capacity", 0.0)),
            }
            for x in sub.itertuples(index=False)
        ]
    return out


def parse_bbox_param(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    parts = [float(x.strip()) for x in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be south,west,north,east")
    return parts[0], parts[1], parts[2], parts[3]
