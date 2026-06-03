# Deploy lgb_fastapi_ui to server

The server only has an **old partial** copy. These files must exist under
`/opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui/` before tract setup runs.

## Quick check on server

```bash
ls -la /opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui/build_census_tract_strata.py
ls -la /opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui/download_tract_shapes.py
```

If either says "No such file", upload first (below).

## Option A — Upload zip (recommended)

**On your Windows PC** (PowerShell), from repo root:

```powershell
cd C:\Users\maple\Desktop\EV_Project\Pro_directory\analysis\gnn_gpu
Compress-Archive -Path lgb_fastapi_ui\* -DestinationPath lgb_fastapi_ui_deploy.zip -Force
scp lgb_fastapi_ui_deploy.zip jovyan@gpuyter.mind.cs.umd.edu:/opt/data_repo/mliang_work/step4_gpu/
```

**On server:**

```bash
cd /opt/data_repo/mliang_work/step4_gpu
unzip -o lgb_fastapi_ui_deploy.zip -d lgb_fastapi_ui
ls lgb_fastapi_ui/build_census_tract_strata.py
```

## Option B — scp whole folder

```powershell
scp -r "C:\Users\maple\Desktop\EV_Project\Pro_directory\analysis\gnn_gpu\lgb_fastapi_ui" `
  jovyan@gpuyter.mind.cs.umd.edu:/opt/data_repo/mliang_work/step4_gpu/
```

## After upload — tract + UI

### Step 1 — strata CSV (needs **CENSUS_API_KEY** on blocked GPU nodes)

Get a free key: https://api.census.gov/data/key_signup.html

```bash
cd /opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui
export DATA_DIR=/opt/data_repo/mliang_work/step4_gpu
export PROJECT_ROOT=/opt/data_repo/mliang_work
export CENSUS_API_KEY='paste_your_key_here'

../.venv_gnn/bin/pip install --no-cache-dir pygris geopandas -q
../.venv_gnn/bin/python build_census_tract_strata.py
```

**Do not** rely on CBSA fallback on the server (`ALLOW_CBSA_FALLBACK=1`) unless
`curl -I https://www2.census.gov` works — most GPU nodes block Census FTP/HTTP.

### Step 2 — tract polygons (often **laptop only**)

On your **Windows PC** (has Census access):

```powershell
cd C:\Users\maple\Desktop\EV_Project\Pro_directory\analysis\gnn_gpu\lgb_fastapi_ui
$env:PROJECT_ROOT="C:\Users\maple\Desktop\EV_Project\Pro_directory"
$env:DATA_DIR="C:\Users\maple\Desktop\EV_Project\Pro_directory\analysis\gnn_gpu"
pip install pygris geopandas
python download_tract_shapes.py
```

Then upload cache to server:

```powershell
scp -r "$env:PROJECT_ROOT\data\processed\tract_shp_2020" `
  jovyan@gpuyter.mind.cs.umd.edu:/opt/data_repo/mliang_work/data/processed/
```

Or zip first if many files.

### Step 3 — UI

```bash
../.venv_gnn/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir .
```

Required files in `lgb_fastapi_ui/`:

- `build_census_tract_strata.py`
- `download_tract_shapes.py`
- `census_tract_strata.py`
- `states.py`
- `project_paths.py`
- `app/` (main.py, tract_boundaries.py, …)
- `static/index.html`
