"""Ad-hoc region prediction: circle, polygon, grid heatmap, county aggregates."""

from __future__ import annotations

import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
MIN_NODES = 15
MAX_NODES = 100


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = np.pi / 180.0
    lat1r, lat2r = lat1 * r, lat2 * r
    dlat = lat2r - lat1r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def point_in_polygon(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray casting; ring = [[lat, lon], ...] closed or open."""
    x, y = lon, lat
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][1], ring[i][0]
        xj, yj = ring[j][1], ring[j][0]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def parse_node_id(node_id: str) -> tuple[str, int]:
    m = re.match(r"^([A-Z]{2})_(\d+)$", str(node_id))
    if not m:
        raise ValueError(f"Bad node_id: {node_id}")
    return m.group(1), int(m.group(2))


class RegionEngine:
    def __init__(self, data_dir: Path, network_dir: Path | None = None) -> None:
        self.data_dir = data_dir.resolve()
        if network_dir is None:
            from network_paths import resolve_network_dir

            network_dir = resolve_network_dir(cwd=self.data_dir)
        self.network_dir = Path(network_dir).resolve()
        self._graph_cache: dict[str, nx.Graph] = {}
        from app.map_geo import load_stations_table

        self._stations = load_stations_table(self.data_dir)

    def _load_graph(self, state: str) -> nx.Graph:
        if state in self._graph_cache:
            return self._graph_cache[state]
        pkl = self.network_dir / f"network_{state}.pkl"
        with open(pkl, "rb") as f:
            payload = pickle.load(f)
        G = payload["network"] if isinstance(payload, dict) else payload
        self._graph_cache[state] = G
        return G

    def nodes_in_circle(self, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
        d = haversine_km(lat, lon, self._stations["lat"].values, self._stations["lon"].values)
        sub = self._stations.loc[d <= radius_km].copy()
        if len(sub) > MAX_NODES:
            sub = sub.sample(n=MAX_NODES, random_state=42)
        return sub

    def nodes_in_polygon(self, ring: list[list[float]]) -> pd.DataFrame:
        if len(ring) < 3:
            raise ValueError("Polygon needs at least 3 vertices")
        mask = [
            point_in_polygon(float(r["lat"]), float(r["lon"]), ring)
            for _, r in self._stations.iterrows()
        ]
        sub = self._stations.loc[mask].copy()
        if len(sub) > MAX_NODES:
            sub = sub.sample(n=MAX_NODES, random_state=42)
        return sub

    def build_induced_graph(self, nodes_df: pd.DataFrame) -> nx.Graph | None:
        G_all = nx.Graph()
        for state, grp in nodes_df.groupby("state"):
            try:
                G_state = self._load_graph(str(state))
            except FileNotFoundError:
                continue
            local_ids = [parse_node_id(nid)[1] for nid in grp["node_id"]]
            local_ids = [i for i in local_ids if i in G_state]
            if not local_ids:
                continue
            sub = G_state.subgraph(local_ids).copy()
            for u, v, data in sub.edges(data=True):
                su = f"{state}_{u}"
                sv = f"{state}_{v}"
                if not G_all.has_node(su):
                    G_all.add_node(su, capacity=float(sub.nodes[u].get("capacity", 0.0)))
                if not G_all.has_node(sv):
                    G_all.add_node(sv, capacity=float(sub.nodes[v].get("capacity", 0.0)))
                w = float(data.get("weight", 1.0))
                G_all.add_edge(su, sv, weight=w)
        if G_all.number_of_nodes() < 2:
            return None
        return G_all

    def graph_to_npz_arrays(self, G: nx.Graph) -> dict[str, np.ndarray]:
        nodes = list(G.nodes())
        mapping = {old: i for i, old in enumerate(nodes)}
        H = nx.relabel_nodes(G, mapping)
        n = H.number_of_nodes()
        caps = np.array([float(H.nodes[i].get("capacity", 0.0)) for i in range(n)], dtype=np.float32)
        edges = list(H.edges(data=True))
        if edges:
            src = np.array([u for u, v, _ in edges], dtype=np.int32)
            dst = np.array([v for u, v, _ in edges], dtype=np.int32)
            w = np.array([float(d.get("weight", 1.0)) for _, _, d in edges], dtype=np.float32)
            edge_index = np.concatenate([np.stack([src, dst], 0), np.stack([dst, src], 0)], axis=1)
            w = np.concatenate([w, w])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int32)
            w = np.zeros(0, dtype=np.float32)
        return {"n_nodes": np.int32(n), "edge_index": edge_index, "edge_weight": w, "capacity": caps}

    def compute_features(
        self,
        nodes_df: pd.DataFrame,
        *,
        fast: bool = True,
        feature_set: str = "extended",
    ) -> dict[str, float]:
        from gnn.global_graph_features import global_features_from_npz, get_global_feature_names

        G = self.build_induced_graph(nodes_df)
        if G is None:
            raise ValueError(f"Could not build graph (need ≥2 connected nodes, got {len(nodes_df)} stations)")
        arrays = self.graph_to_npz_arrays(G)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            path = Path(tmp.name)
            np.savez_compressed(path, **arrays)
        try:
            feats = global_features_from_npz(path, fast=fast, feature_set=feature_set)
        finally:
            path.unlink(missing_ok=True)
        names = get_global_feature_names(feature_set)
        return {k: float(feats[k]) for k in names}

    @staticmethod
    def stations_list(nodes_df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for r in nodes_df.itertuples(index=False):
            rows.append(
                {
                    "node_id": str(r.node_id),
                    "lat": float(r.lat),
                    "lon": float(r.lon),
                    "state": str(getattr(r, "state", "")),
                    "capacity": float(getattr(r, "capacity", 0.0)),
                    "stratum": str(getattr(r, "node_stratum", getattr(r, "stratum", ""))),
                }
            )
        return rows

    def region_summary(self, nodes_df: pd.DataFrame, center_lat: float, center_lon: float, radius_km: float | None) -> dict[str, Any]:
        return {
            "n_stations": int(len(nodes_df)),
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "radius_km": float(radius_km) if radius_km is not None else None,
            "states": sorted(nodes_df["state"].astype(str).unique().tolist()) if "state" in nodes_df.columns else [],
            "stations": self.stations_list(nodes_df),
        }

    def predict_circle(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        *,
        fast: bool = True,
        feature_set: str = "extended",
    ) -> dict[str, Any]:
        nodes_df = self.nodes_in_circle(lat, lon, radius_km)
        if len(nodes_df) < MIN_NODES:
            raise ValueError(f"Only {len(nodes_df)} stations in circle (need ≥{MIN_NODES})")
        feats = self.compute_features(nodes_df, fast=fast, feature_set=feature_set)
        out = self.region_summary(nodes_df, lat, lon, radius_km)
        out["features"] = feats
        out["region_type"] = "circle"
        return out

    def predict_polygon(
        self,
        ring: list[list[float]],
        *,
        fast: bool = True,
        feature_set: str = "extended",
    ) -> dict[str, Any]:
        nodes_df = self.nodes_in_polygon(ring)
        if len(nodes_df) < MIN_NODES:
            raise ValueError(f"Only {len(nodes_df)} stations in polygon (need ≥{MIN_NODES})")
        lat = float(nodes_df["lat"].mean())
        lon = float(nodes_df["lon"].mean())
        feats = self.compute_features(nodes_df, fast=fast, feature_set=feature_set)
        out = self.region_summary(nodes_df, lat, lon, None)
        out["features"] = feats
        out["region_type"] = "polygon"
        out["polygon"] = ring
        return out

    def predict_heatmap_grid(
        self,
        *,
        south: float,
        west: float,
        north: float,
        east: float,
        step_deg: float = 0.75,
        radius_km: float = 30.0,
        max_cells: int = 250,
        fast: bool = True,
        feature_set: str = "extended",
    ) -> dict[str, Any]:
        lats = np.arange(south, north + 1e-9, step_deg)
        lons = np.arange(west, east + 1e-9, step_deg)
        grid = [(float(la), float(lo)) for la in lats for lo in lons]
        if len(grid) > max_cells:
            idx = np.linspace(0, len(grid) - 1, max_cells, dtype=int)
            grid = [grid[i] for i in idx]
        cells: list[dict[str, Any]] = []
        errors = 0
        for lat, lon in grid:
            try:
                nodes_df = self.nodes_in_circle(lat, lon, radius_km)
                if len(nodes_df) < MIN_NODES:
                    errors += 1
                    continue
                feats = self.compute_features(nodes_df, fast=fast, feature_set=feature_set)
                cells.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "radius_km": radius_km,
                        "n_stations": int(len(nodes_df)),
                        "features": feats,
                    }
                )
            except Exception:
                errors += 1
        return {
            "n_cells": len(cells),
            "n_skipped": errors,
            "step_deg": step_deg,
            "radius_km": radius_km,
            "cells": cells,
        }


def county_aggregates(
    data_dir: Path,
    map_df: pd.DataFrame | None,
    nodes: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Aggregate window predictions by county FIPS (first 5 digits of tract GEOID)."""
    meta_path = data_dir / "windows_meta.csv"
    if not meta_path.is_file():
        return []
    meta = pd.read_csv(meta_path)
    df = meta[["window_id", "center_lat", "center_lon"]].copy()
    if map_df is not None and "pred_mean" in map_df.columns:
        pred = map_df[["window_id", "pred_mean", "pred_peak"]].drop_duplicates("window_id")
        df = df.merge(pred, on="window_id", how="left")
    else:
        df["pred_mean"] = np.nan
        df["pred_peak"] = np.nan

    county_col = None
    if nodes is not None and "tract_geoid" in nodes.columns:
        tg = nodes.copy()
        tg["tract_geoid"] = tg["tract_geoid"].astype(str)
        tg["county_fips"] = tg["tract_geoid"].str.slice(0, 5)
        tg = tg[tg["county_fips"].str.len() == 5]
        mode_county = (
            tg.groupby("window_id")["county_fips"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
            .reset_index()
        )
        df = df.merge(mode_county, on="window_id", how="left")
        county_col = "county_fips"

    if county_col is None or df[county_col].isna().all():
        df["county_fips"] = (df["center_lat"].round(1).astype(str) + "_" + df["center_lon"].round(1).astype(str))
        county_col = "county_fips"

    rows = []
    for fips, grp in df.groupby(county_col):
        if not str(fips).strip():
            continue
        rows.append(
            {
                "county_fips": str(fips),
                "n_windows": int(len(grp)),
                "pred_mean_avg": float(grp["pred_mean"].mean()) if grp["pred_mean"].notna().any() else None,
                "pred_peak_max": float(grp["pred_peak"].max()) if grp.get("pred_peak") is not None and grp["pred_peak"].notna().any() else None,
                "center_lat": float(grp["center_lat"].mean()),
                "center_lon": float(grp["center_lon"].mean()),
            }
        )
    rows.sort(key=lambda r: (r["pred_mean_avg"] is None, -(r["pred_mean_avg"] or 0)))
    return rows
