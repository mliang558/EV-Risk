"""Resolve repo / data root on laptop or server (step4_gpu only)."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(data_dir: Path | None = None) -> Path:
    """
    Where ``data/processed/`` lives.

    Server (typical):
      export DATA_DIR=/opt/data_repo/mliang_work/step4_gpu
      export PROJECT_ROOT=/opt/data_repo/mliang_work   # optional, recommended
    """
    env_root = os.environ.get("PROJECT_ROOT", "").strip()
    if env_root:
        root = Path(env_root).resolve()
        (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
        return root

    if data_dir is not None:
        data_dir = Path(data_dir).resolve()
        for cand in (data_dir.parent.parent, data_dir.parent, data_dir):
            if cand.is_dir():
                (cand / "data" / "processed").mkdir(parents=True, exist_ok=True)
                return cand

    ui = Path(__file__).resolve().parent
    for cand in (ui.parent.parent.parent, ui.parent.parent, ui.parent):
        if cand.is_dir():
            (cand / "data" / "processed").mkdir(parents=True, exist_ok=True)
            return cand
    return ui.parent.parent.parent
