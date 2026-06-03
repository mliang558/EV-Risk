#!/usr/bin/env python3
"""
One-time: download 2020 census tract polygons for all EV states (pygris).

Run on server from lgb_fastapi_ui (no parent analysis/ tree required):

  cd /opt/data_repo/mliang_work/step4_gpu/lgb_fastapi_ui
  export DATA_DIR=/opt/data_repo/mliang_work/step4_gpu
  export PROJECT_ROOT=/opt/data_repo/mliang_work
  ../.venv_gnn/bin/pip install pygris geopandas -q
  ../.venv_gnn/bin/python build_census_tract_strata.py
  ../.venv_gnn/bin/python download_tract_shapes.py

Output: $PROJECT_ROOT/data/processed/tract_shp_2020/{ST}_tract2020.parquet
Needs: data/processed/census_tract_strata.csv (from build_census_tract_strata.py)
"""

from __future__ import annotations

import os
from pathlib import Path

from census_tract_strata import load_or_build_tract_strata, load_state_tracts, tract_shp_cache_dir
from project_paths import resolve_project_root
from states import STATES_ORDER


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent)).resolve()
    project_root = resolve_project_root(data_dir)
    print(f"Project root: {project_root}", flush=True)
    load_or_build_tract_strata(project_root)
    shp_dir = tract_shp_cache_dir(project_root)
    print(f"Tract cache dir: {shp_dir}", flush=True)
    for i, abbr in enumerate(STATES_ORDER, 1):
        print(f"[{i}/{len(STATES_ORDER)}] {abbr} ...", flush=True)
        load_state_tracts(abbr, project_root)
    print("Done. Restart UI and enable Stratum boundaries.", flush=True)


if __name__ == "__main__":
    main()
