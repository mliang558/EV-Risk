from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from app.map_geo import (
    STRATUM_RADIUS_KM,
    classify_location,
    load_stations_table,
    load_windows_meta,
    parse_bbox_param,
    stations_payload,
    stratum_legend_payload,
    window_detail,
    windows_payload,
)
from app.region_predict import RegionEngine, county_aggregates
from app.tract_boundaries import tract_strata_geojson


# Make gnn_gpu scripts importable.
GNN_GPU_ROOT = Path(__file__).resolve().parents[2]
if str(GNN_GPU_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_GPU_ROOT))
_ANALYSIS_ROOT = GNN_GPU_ROOT.parent
if _ANALYSIS_ROOT.is_dir() and str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from gnn.constants import N_CURVE_POINTS, parse_attacks, y_cols_for_attacks
from gnn.metrics import metrics_by_attack
from step7_window_lgb_baseline import build_feature_table, make_model


APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = APP_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(GNN_GPU_ROOT))).resolve()


class TrainRequest(BaseModel):
    data_dir: str = "."
    windows_meta: str = "windows_meta.csv"
    labels: str | None = None
    lon_threshold: float = -100.0
    attacks: str = "no-capacity"
    model: str = "lightgbm"
    n_estimators: int = 200
    max_depth: int = 8
    num_leaves: int = 31
    n_jobs: int = -1
    global_feature_set: str = "extended"
    global_fast: bool = False
    reuse_features: bool = True


class PredictRequest(BaseModel):
    window_id: str = Field(..., description="window_id from windows_meta")


class PredictCircleRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(30.0, gt=0, le=200)
    fast: bool = True


class PredictPolygonRequest(BaseModel):
    ring: list[list[float]] = Field(..., min_length=3, description="[[lat, lon], ...]")
    fast: bool = True


class PredictHeatmapRequest(BaseModel):
    south: float
    west: float
    north: float
    east: float
    step_deg: float = 0.75
    radius_km: float = 30.0
    max_cells: int = 200
    fast: bool = True


class MapDataCache:
    """Static map layers from Step-3/4 (stations + window disks)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.ready = False
        self.error: str | None = None
        self.stations: pd.DataFrame | None = None
        self.meta: pd.DataFrame | None = None
        self.nodes: pd.DataFrame | None = None

    def load(self) -> bool:
        try:
            self.stations = load_stations_table(self.data_dir)
            self.meta = load_windows_meta(self.data_dir)
            nodes_path = self.data_dir / "windows_nodes.parquet"
            self.nodes = pd.read_parquet(nodes_path) if nodes_path.is_file() else None
            self.ready = True
            self.error = None
            return True
        except Exception as e:
            self.ready = False
            self.error = str(e)
            return False

    def pred_lookup(self, service: "LGBService") -> dict[str, dict[str, float]]:
        if not service.ready or service.map_df is None:
            return {}
        out: dict[str, dict[str, float]] = {}
        for _, r in service.map_df.iterrows():
            wid = str(r["window_id"])
            out[wid] = {
                "pred_mean": float(r.get("pred_mean", np.nan)),
                "pred_peak": float(r.get("pred_peak", np.nan)),
                "true_mean": float(r.get("true_mean", np.nan)),
            }
        return out


class LGBService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.meta: dict[str, Any] = {}
        self.model = None
        self.scaler = None
        self.feature_df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.pred_df: pd.DataFrame | None = None
        self.map_df: pd.DataFrame | None = None

    def _artifacts(self) -> dict[str, Path]:
        return {
            "model": ARTIFACT_DIR / "lgb_model.joblib",
            "scaler": ARTIFACT_DIR / "scaler.joblib",
            "meta": ARTIFACT_DIR / "meta.json",
            "train_df": ARTIFACT_DIR / "train_merged.parquet",
            "pred_df": ARTIFACT_DIR / "west_predictions.parquet",
            "map_df": ARTIFACT_DIR / "map_points.parquet",
        }

    def _paths_in_dir(self, artifact_dir: Path) -> dict[str, Path]:
        d = Path(artifact_dir).resolve()
        return {
            "model": d / "lgb_model.joblib",
            "scaler": d / "scaler.joblib",
            "meta": d / "meta.json",
            "train_df": d / "train_merged.parquet",
            "pred_df": d / "west_predictions.parquet",
            "map_df": d / "map_points.parquet",
        }

    def load_from_dir(self, artifact_dir: Path) -> bool:
        """Load joblib + parquet bundle from a directory (same layout as ARTIFACT_DIR)."""
        paths = self._paths_in_dir(artifact_dir)
        required = ("model", "scaler", "meta", "train_df", "pred_df")
        if not all(paths[k].exists() for k in required):
            return False
        self.model = joblib.load(paths["model"])
        self.scaler = joblib.load(paths["scaler"])
        self.meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.train_df = pd.read_parquet(paths["train_df"])
        self.pred_df = pd.read_parquet(paths["pred_df"])
        if paths["map_df"].is_file():
            self.map_df = pd.read_parquet(paths["map_df"])
        else:
            self.map_df = self.pred_df.copy()
        self.ready = True
        return True

    def load_if_exists(self) -> bool:
        for cand in [ARTIFACT_DIR]:
            if self.load_from_dir(cand):
                return True
        extra = os.environ.get("LGB_UI_ARTIFACT_DIR", "").strip()
        if extra and self.load_from_dir(Path(extra)):
            return True
        for cand in (
            GNN_GPU_ROOT / "lgb_fastapi_ui" / "artifacts",
            DEFAULT_DATA_DIR / "lgb_fastapi_ui" / "artifacts",
        ):
            if cand.is_dir() and self.load_from_dir(cand):
                return True
        return False

    def train(self, req: TrainRequest) -> dict[str, Any]:
        with self.lock:
            data_dir = Path(req.data_dir).resolve()
            if str(req.data_dir) in (".", "./"):
                data_dir = GNN_GPU_ROOT.resolve()

            attacks = parse_attacks(req.attacks)
            labels_path = Path(req.labels) if req.labels else data_dir / "labels" / "y_labels.parquet"
            if not labels_path.exists():
                labels_path = data_dir / "labels" / "y_labels.csv"
            if not labels_path.exists():
                raise HTTPException(status_code=400, detail=f"Labels file not found: {labels_path}")

            windows_meta_path = data_dir / req.windows_meta
            if not windows_meta_path.exists():
                raise HTTPException(status_code=400, detail=f"windows_meta not found: {windows_meta_path}")

            meta_df = pd.read_csv(windows_meta_path)
            labels = (
                pd.read_parquet(labels_path)
                if labels_path.suffix == ".parquet"
                else pd.read_csv(labels_path)
            )
            merged = labels.copy()
            meta_cols = [
                c
                for c in ("window_id", "center_lon", "center_lat", "window_stratum", "radius_km", "n_nodes")
                if c in meta_df.columns
            ]
            if len(meta_cols) > 1:
                merged = merged.merge(meta_df[meta_cols], on="window_id", how="left", suffixes=("", "_meta"))

            cache_features = ARTIFACT_DIR / "window_global_features.csv"
            global_df = build_feature_table(
                merged,
                data_dir,
                cache_features,
                reuse_only=req.reuse_features,
                feature_set=req.global_feature_set,
                fast=req.global_fast,
            )
            overlap = [c for c in global_df.columns if c in merged.columns and c != "window_id"]
            merged = merged.merge(global_df.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")

            x_cols = [c for c in global_df.columns if c != "window_id"]
            y_cols = [c for c in y_cols_for_attacks(attacks) if c in merged.columns]
            expected_targets = len(attacks) * N_CURVE_POINTS
            if len(y_cols) != expected_targets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need {expected_targets} target columns for attacks={attacks}, got {len(y_cols)}",
                )

            X = merged[x_cols].astype(float).values
            Y = merged[y_cols].astype(float).values
            train_mask = merged["center_lon"].astype(float).values > req.lon_threshold
            test_mask = ~train_mask

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_test = scaler.transform(X[test_mask])
            Y_train, Y_test = Y[train_mask], Y[test_mask]

            model = make_model(req.model, req.n_estimators, req.max_depth, req.num_leaves, req.n_jobs)
            model.fit(pd.DataFrame(X_train, columns=x_cols), Y_train)

            pred_train = model.predict(pd.DataFrame(X_train, columns=x_cols))
            pred_test = model.predict(pd.DataFrame(X_test, columns=x_cols))

            overall_train_r2 = float(r2_score(Y_train.ravel(), pred_train.ravel()))
            overall_test_r2 = float(r2_score(Y_test.ravel(), pred_test.ravel()))
            overall_test_mae = float(mean_absolute_error(Y_test.ravel(), pred_test.ravel()))
            by_attack_test = metrics_by_attack(Y_test, pred_test, attacks=attacks, n_points=N_CURVE_POINTS)
            by_attack_train = metrics_by_attack(Y_train, pred_train, attacks=attacks, n_points=N_CURVE_POINTS)

            test_rows = merged.loc[
                test_mask, ["window_id", "window_stratum", "radius_km", "center_lon", "center_lat", "n_nodes"]
            ].copy()
            test_rows["pred_mean"] = pred_test.mean(axis=1)
            test_rows["pred_peak"] = pred_test.max(axis=1)
            test_rows["true_mean"] = Y_test.mean(axis=1)
            test_rows["true_peak"] = Y_test.max(axis=1)
            test_rows["abs_err_mean"] = np.abs(test_rows["pred_mean"] - test_rows["true_mean"])
            test_rows = test_rows.sort_values("pred_mean", ascending=False).reset_index(drop=True)

            east_rows = merged.loc[
                train_mask, ["window_id", "window_stratum", "radius_km", "center_lon", "center_lat", "n_nodes"]
            ].copy()
            east_rows["pred_mean"] = pred_train.mean(axis=1)
            east_rows["pred_peak"] = pred_train.max(axis=1)
            east_rows["true_mean"] = Y_train.mean(axis=1)
            east_rows["split"] = "east"
            test_rows["split"] = "west"
            self.map_df = pd.concat([test_rows, east_rows], ignore_index=True)

            self.model = model
            self.scaler = scaler
            self.feature_df = global_df
            self.train_df = merged[["window_id", *x_cols, *y_cols, "center_lon"]].copy()
            self.pred_df = test_rows
            self.meta = {
                "attacks": list(attacks),
                "x_cols": x_cols,
                "y_cols": y_cols,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_r2": overall_train_r2,
                "test_r2": overall_test_r2,
                "test_mae": overall_test_mae,
                "by_attack_test": {k: {"r2": float(v["r2"]), "mae": float(v["mae"])} for k, v in by_attack_test.items()},
                "by_attack_train": {k: {"r2": float(v["r2"]), "mae": float(v["mae"])} for k, v in by_attack_train.items()},
                "config": req.model_dump(),
            }
            self.ready = True

            paths = self._artifacts()
            joblib.dump(self.model, paths["model"])
            joblib.dump(self.scaler, paths["scaler"])
            paths["meta"].write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
            self.train_df.to_parquet(paths["train_df"], index=False)
            self.pred_df.to_parquet(paths["pred_df"], index=False)
            self.map_df.to_parquet(paths["map_df"], index=False)

            return self.meta

    def predict_window(self, window_id: str) -> dict[str, Any]:
        if not self.ready or self.train_df is None:
            raise HTTPException(status_code=400, detail="Model not trained/loaded yet.")
        row = self.train_df[self.train_df["window_id"].astype(str) == str(window_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"window_id not found: {window_id}")
        x_cols = self.meta["x_cols"]
        attacks = tuple(self.meta["attacks"])
        x = row.iloc[0][x_cols].astype(float).values.reshape(1, -1)
        x = self.scaler.transform(x)
        pred = self.model.predict(pd.DataFrame(x, columns=x_cols))[0]
        curves_loss: dict[str, list[float]] = {}
        for i, atk in enumerate(attacks):
            s = i * N_CURVE_POINTS
            e = (i + 1) * N_CURVE_POINTS
            curves_loss[atk] = [float(v) for v in pred[s:e]]
        curves_perf = {atk: [float(1.0 - v) for v in vals] for atk, vals in curves_loss.items()}
        mean_loss = float(np.mean(pred))
        return {
            "window_id": str(window_id),
            "pred_curves": curves_loss,
            "pred_curves_performance": curves_perf,
            "pred_mean": mean_loss,
            "pred_mean_loss": mean_loss,
            "pred_mean_performance": float(1.0 - mean_loss),
        }

    def top_windows(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.ready or self.pred_df is None:
            raise HTTPException(status_code=400, detail="Model not trained/loaded yet.")
        limit = max(1, min(limit, 200))
        out = self.pred_df.head(limit).copy()
        return out.to_dict(orient="records")

    def predict_from_features(self, features: dict[str, float]) -> dict[str, Any]:
        if not self.ready:
            raise HTTPException(status_code=400, detail="Model not trained/loaded yet.")
        x_cols = self.meta["x_cols"]
        attacks = tuple(self.meta["attacks"])
        missing = [c for c in x_cols if c not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing[:5]}")
        x = np.array([[float(features[c]) for c in x_cols]], dtype=np.float64)
        x = self.scaler.transform(x)
        pred = self.model.predict(pd.DataFrame(x, columns=x_cols))[0]
        curves: dict[str, list[float]] = {}
        for i, atk in enumerate(attacks):
            s = i * N_CURVE_POINTS
            e = (i + 1) * N_CURVE_POINTS
            curves[atk] = [float(v) for v in pred[s:e]]
        return {
            "pred_curves": curves,
            "pred_mean": float(np.mean(pred)),
            "pred_peak": float(np.max(pred)),
        }


service = LGBService()
map_cache = MapDataCache(DEFAULT_DATA_DIR)
region_engine: RegionEngine | None = None
region_engine_error: str | None = None
app = FastAPI(title="EV Window Explorer API", version="2.1.0")


def _require_region() -> RegionEngine:
    if region_engine is None:
        raise HTTPException(
            status_code=503,
            detail=region_engine_error or "Region engine not available (need network_structures/)",
        )
    return region_engine


def _attach_prediction(region_out: dict[str, Any]) -> dict[str, Any]:
    pred = service.predict_from_features(region_out["features"])
    region_out.update(pred)
    if "pred_curves" in region_out:
        region_out["pred_curves_performance"] = {
            atk: [float(1.0 - v) for v in vals] for atk, vals in region_out["pred_curves"].items()
        }
    if "pred_mean" in region_out:
        region_out["pred_mean_loss"] = float(region_out["pred_mean"])
        region_out["pred_mean_performance"] = float(1.0 - region_out["pred_mean_loss"])
    return region_out


def _bootstrap_model_if_needed() -> None:
    if service.ready:
        return
    if os.environ.get("LGB_AUTO_BOOTSTRAP", "").strip() in ("1", "true", "yes"):
        labels = DEFAULT_DATA_DIR / "labels" / "y_labels.parquet"
        if labels.is_file():
            try:
                service.train(TrainRequest(data_dir=str(DEFAULT_DATA_DIR), reuse_features=True))
                return
            except Exception as e:
                print(f"[UI] Auto-bootstrap train failed: {e}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    global region_engine, region_engine_error
    service.load_if_exists()
    _bootstrap_model_if_needed()
    map_cache.load()
    try:
        region_engine = RegionEngine(map_cache.data_dir)
        region_engine_error = None
    except Exception as e:
        region_engine = None
        region_engine_error = str(e)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    html_path = APP_ROOT / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


def _tract_data_ready(data_dir: Path) -> tuple[bool, str | None]:
    root = data_dir.parent.parent
    for base in (root, data_dir.parent, GNN_GPU_ROOT.parent.parent):
        strata = base / "data" / "processed" / "census_tract_strata.csv"
        shp = base / "data" / "processed" / "tract_shp_2020"
        if strata.is_file() and shp.is_dir() and any(shp.glob("*_tract2020.parquet")):
            return True, str(base)
    return False, None


@app.get("/api/health")
def health() -> dict[str, Any]:
    tract_ok, tract_root = _tract_data_ready(map_cache.data_dir)
    return {
        "ok": True,
        "ready": service.ready,
        "map_ready": map_cache.ready,
        "map_error": map_cache.error,
        "data_dir": str(map_cache.data_dir),
        "region_ready": region_engine is not None,
        "region_error": region_engine_error,
        "tract_boundaries_ready": tract_ok,
        "tract_data_root": tract_root,
        "stratum_radii_km": STRATUM_RADIUS_KM,
    }


@app.get("/api/map/stratum-legend")
def map_stratum_legend() -> dict[str, Any]:
    return stratum_legend_payload()


@app.get("/api/map/tract-strata")
def map_tract_strata(
    bbox: str = Query(..., description="south,west,north,east"),
    max_features: int = Query(2000, ge=100, le=8000),
    simplify: float = Query(0.008, ge=0.0, le=0.05),
) -> dict[str, Any]:
    """Census tract polygons colored by urban / suburban / rural (map boundaries, not window disks)."""
    try:
        bb = parse_bbox_param(bbox)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if bb is None:
        raise HTTPException(status_code=400, detail="bbox required")
    return tract_strata_geojson(
        map_cache.data_dir,
        bb,
        max_features=max_features,
        simplify_tol=simplify,
    )


@app.get("/api/map/stations-in-circle")
def map_stations_in_circle(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(30.0, gt=0, le=200),
) -> dict[str, Any]:
    """Stations inside a circle (full national station table, for map highlight)."""
    from app.region_predict import RegionEngine

    if not map_cache.ready or map_cache.stations is None:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    st = map_cache.stations
    from app.region_predict import haversine_km

    d = haversine_km(lat, lon, st["lat"].values.astype(float), st["lon"].values.astype(float))
    nodes_df = st.loc[d <= float(radius_km)].copy()
    return {
        "n_stations": int(len(nodes_df)),
        "center_lat": lat,
        "center_lon": lon,
        "radius_km": float(radius_km),
        "stations": RegionEngine.stations_list(nodes_df),
    }


@app.get("/api/map/probe")
def map_probe(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    if not map_cache.ready or map_cache.stations is None:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    from app.tract_boundaries import resolve_project_root

    root = resolve_project_root(map_cache.data_dir)
    info = classify_location(
        lat, lon, stations=map_cache.stations, meta=map_cache.meta, project_root=root
    )
    return info


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    if not service.ready:
        raise HTTPException(status_code=400, detail="Model not ready.")
    return service.meta


@app.post("/api/train")
def train(req: TrainRequest) -> dict[str, Any]:
    return service.train(req)


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    return service.predict_window(req.window_id)


@app.get("/api/windows/top")
def top_windows(limit: int = 30) -> dict[str, Any]:
    return {"items": service.top_windows(limit=limit)}


@app.get("/api/map/stations")
def map_stations(
    bbox: str | None = Query(None, description="south,west,north,east"),
    limit: int = Query(20000, ge=100, le=50000),
    stratum: str | None = None,
) -> dict[str, Any]:
    if not map_cache.ready or map_cache.stations is None:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    return stations_payload(
        map_cache.stations,
        bbox=parse_bbox_param(bbox),
        limit=limit,
        stratum=stratum,
    )


@app.get("/api/map/windows")
def map_windows(
    bbox: str | None = Query(None),
    limit: int = Query(5000, ge=50, le=10000),
    window_stratum: str | None = None,
    radius_km: float | None = None,
) -> dict[str, Any]:
    if not map_cache.ready or map_cache.meta is None:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    return windows_payload(
        map_cache.meta,
        map_cache.pred_lookup(service),
        bbox=parse_bbox_param(bbox),
        limit=limit,
        window_stratum=window_stratum,
        radius_km=radius_km,
    )


@app.get("/api/map/window/{window_id}")
def map_window_one(window_id: str) -> dict[str, Any]:
    if not map_cache.ready or map_cache.meta is None:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    try:
        detail = window_detail(
            map_cache.data_dir,
            window_id,
            map_cache.meta,
            map_cache.nodes,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"window not found: {window_id}") from None
    preds = map_cache.pred_lookup(service).get(str(window_id))
    if preds:
        detail["predictions"] = preds
    return detail


def _predict_circle(lat: float, lon: float, radius_km: float, fast: bool) -> dict[str, Any]:
    eng = _require_region()
    if not service.ready:
        raise HTTPException(status_code=400, detail="Train or load model first.")
    try:
        out = eng.predict_circle(lat, lon, radius_km, fast=fast)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _attach_prediction(out)


@app.post("/api/predict/circle")
def predict_circle_post(body: PredictCircleRequest = Body(...)) -> dict[str, Any]:
    return _predict_circle(body.lat, body.lon, body.radius_km, body.fast)


@app.get("/api/predict/circle")
def predict_circle_get(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(30.0, gt=0, le=200),
    fast: bool = Query(True),
) -> dict[str, Any]:
    """Query-param fallback when POST JSON body is not accepted (older proxies / misconfigured routes)."""
    return _predict_circle(lat, lon, radius_km, fast)


@app.post("/api/predict/polygon")
def predict_polygon(req: PredictPolygonRequest) -> dict[str, Any]:
    eng = _require_region()
    if not service.ready:
        raise HTTPException(status_code=400, detail="Train or load model first.")
    try:
        out = eng.predict_polygon(req.ring, fast=req.fast)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _attach_prediction(out)


@app.post("/api/predict/heatmap")
def predict_heatmap(req: PredictHeatmapRequest) -> dict[str, Any]:
    eng = _require_region()
    if not service.ready:
        raise HTTPException(status_code=400, detail="Train or load model first.")
    grid = eng.predict_heatmap_grid(
        south=req.south,
        west=req.west,
        north=req.north,
        east=req.east,
        step_deg=req.step_deg,
        radius_km=req.radius_km,
        max_cells=min(req.max_cells, 400),
        fast=req.fast,
    )
    x_cols = service.meta["x_cols"]
    points = []
    for cell in grid["cells"]:
        try:
            pred = service.predict_from_features(cell["features"])
            points.append(
                {
                    "lat": cell["lat"],
                    "lon": cell["lon"],
                    "n_stations": cell["n_stations"],
                    "pred_mean": pred["pred_mean"],
                    "pred_peak": pred["pred_peak"],
                }
            )
        except Exception:
            continue
    vals = [p["pred_mean"] for p in points]
    return {
        "n_points": len(points),
        "n_skipped": grid["n_skipped"] + (len(grid["cells"]) - len(points)),
        "step_deg": req.step_deg,
        "radius_km": req.radius_km,
        "value_min": float(min(vals)) if vals else None,
        "value_max": float(max(vals)) if vals else None,
        "points": points,
    }


@app.get("/api/map/counties")
def map_counties() -> dict[str, Any]:
    if not map_cache.ready:
        raise HTTPException(status_code=503, detail=map_cache.error or "Map data not loaded")
    items = county_aggregates(
        map_cache.data_dir,
        service.map_df if service.ready else None,
        map_cache.nodes,
    )
    vals = [x["pred_mean_avg"] for x in items if x.get("pred_mean_avg") is not None]
    return {
        "n_counties": len(items),
        "value_min": float(min(vals)) if vals else None,
        "value_max": float(max(vals)) if vals else None,
        "counties": items,
    }
