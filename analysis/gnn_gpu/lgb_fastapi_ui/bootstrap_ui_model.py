#!/usr/bin/env python3
"""One-shot: train LightGBM on DATA_DIR and save artifacts/ for the FastAPI UI (no UI train click)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_GNN = _ROOT.parent
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from app.main import ARTIFACT_DIR, TrainRequest, service  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", str(_GNN))).resolve()
    print(f"Training UI model on {data_dir} -> {ARTIFACT_DIR}", flush=True)
    meta = service.train(
        TrainRequest(
            data_dir=str(data_dir),
            reuse_features=True,
            attacks="no-capacity",
            global_feature_set="extended",
        )
    )
    print(f"Done. test_r2={meta.get('test_r2'):.4f}  artifacts={ARTIFACT_DIR}", flush=True)


if __name__ == "__main__":
    main()
