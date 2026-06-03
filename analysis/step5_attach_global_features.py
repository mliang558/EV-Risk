#!/usr/bin/env python3
"""
Attach standardized global graph features to an existing pyg_dataset.pt (optional).

Most users can skip this and use step6 --global-features --data-dir . instead.

Usage:
  PYTHONPATH=. python step5_attach_global_features.py \\
    --dataset pyg/pyg_dataset.pt --data-dir . \\
    --out pyg/pyg_dataset_global.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.dataset import save_dataset_bundle
from gnn.global_graph_features import attach_global_features_to_graphs


def load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out", default="pyg/pyg_dataset_global.pt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    bundle = load_bundle(Path(args.dataset).resolve())
    graphs = bundle["graphs"]
    train_mask = bundle.get("split_masks", {}).get("east_west_train")
    stats = dict(bundle.get("stats", {}))
    stats.update(attach_global_features_to_graphs(graphs, data_dir, train_mask))

    out = Path(args.out).resolve()
    path = save_dataset_bundle(
        out.parent,
        graphs,
        bundle["meta"],
        stats,
        bundle.get("split_masks"),
    )
    if path != out and out.name != "pyg_dataset.pt":
        import shutil

        shutil.copy2(path, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
