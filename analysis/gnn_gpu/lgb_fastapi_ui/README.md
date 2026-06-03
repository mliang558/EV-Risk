# EV Window Explorer UI

Map-first dashboard for **sliding-window** sampling (not center-point dots).

## Layout

- **Top / main**: Leaflet map
  - Gray dots: clustered charging stations (`windows_nodes.parquet`, unique `node_id`)
  - Semi-transparent **disks** = sliding windows (radius 10 / 30 / 80 km); **no center marker**
  - Click disk → highlight (orange) + stations inside window + predict curves in sidebar
- **Right sidebar**: train, metrics, selected-window analysis

## Data required (in `DATA_DIR`, default parent `gnn_gpu` / `step4_gpu`)

- `windows_meta.csv`
- `windows_nodes.parquet`

## Run

```bash
cd analysis/gnn_gpu/lgb_fastapi_ui
export DATA_DIR=/opt/data_repo/mliang_work/step4_gpu   # or ..
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir .
# JupyterHub: open /user/<you>/proxy/8000/
```

## Census tract shading (urban / suburban / rural polygons)

Upload the whole `lgb_fastapi_ui/` folder (includes tract scripts). On server:

```bash
cd /opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui
export DATA_DIR=/opt/data_repo/mliang_work/step4_gpu
export PROJECT_ROOT=/opt/data_repo/mliang_work

../.venv_gnn/bin/pip install pygris geopandas -q
../.venv_gnn/bin/python build_census_tract_strata.py    # ~10–60 min (CBSA fallback if no CENSUS_API_KEY)
../.venv_gnn/bin/python download_tract_shapes.py        # ~30–60 min, all states

../.venv_gnn/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir .
```

Check `GET /api/health` → `tract_boundaries_ready: true`. Map zoom ≥ 5, enable **Stratum boundaries**.

Do **not** use `python .../analysis/build_census_tract_strata.py` unless you also uploaded the full `analysis/` tree under `/opt/data_repo/mliang_work/`.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/map/stations?bbox=s,w,n,e` | Station points in viewport |
| `GET /api/map/windows?bbox=...` | Window disks + optional pred coloring |
| `GET /api/map/window/{id}` | Window meta + member stations |
| `POST /api/predict` | LGB curves for selected window |

Map reloads on pan/zoom (viewport bbox). Use stratum filter for urban/suburban/rural only.

## Custom regions (v2.1)

Requires `network_structures/network_*.pkl` under `DATA_DIR` and trained `artifacts/`.

| Feature | UI |
|---------|-----|
| Circle + radius | **Draw circle** on map → instant predict |
| Service territory | **Draw territory** (polygon) |
| National heatmap | **Heatmap** (current viewport grid) |
| County aggregate | **Counties** checkbox |

API: `POST /api/predict/circle`, `/polygon`, `/heatmap`; `GET /api/map/counties`.
