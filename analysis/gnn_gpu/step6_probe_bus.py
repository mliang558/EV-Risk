#!/usr/bin/env python3
"""
Pinpoint Step-6 Bus error: run phases one-by-one (each prints before work).

  cd step4_gpu
  export MPLCONFIGDIR=/tmp/mpl_$USER
  PYTHONPATH=. python -u step6_probe_bus.py --dataset pyg/pyg_dataset.pt
  PYTHONPATH=. python -u step6_probe_bus.py --dataset pyg/pyg_dataset.pt \\
      --global-features-csv results_gnn/xgb_baseline/window_global_features.csv \\
      --device cuda --forward
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def phase(name: str) -> None:
    print(f"\n=== PHASE: {name} ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--global-features-csv", default=None)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--forward", action="store_true", help="Run one GAT+global forward on device")
    parser.add_argument("--sanitize", action="store_true", help="Run sanitize_graph_list")
    args = parser.parse_args()

    phase("env")
    print(f"pid={os.getpid()} cwd={Path.cwd()}", flush=True)
    try:
        import shutil

        total, used, free = shutil.disk_usage(Path.cwd())
        print(f"disk_free_gb={free / (1024**3):.2f} total_gb={total / (1024**3):.1f}", flush=True)
        if free < 500 * 1024 * 1024:
            print("WARN: <500MB free — torch.load/save often Bus-errors", flush=True)
    except Exception as e:
        print(f"disk check skipped: {e}", flush=True)

    phase("import torch")
    import torch

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

    phase("import pyg")
    from torch_geometric.loader import DataLoader

    from gnn.edge_utils import sanitize_graph_list
    from gnn.models import ResilienceGNN

    ds_path = Path(args.dataset).resolve()
    phase(f"torch.load {ds_path}")
    t0 = time.perf_counter()
    try:
        bundle = torch.load(ds_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(ds_path, map_location="cpu")
    graphs = bundle["graphs"]
    print(f"loaded n={len(graphs)} in {time.perf_counter() - t0:.1f}s", flush=True)
    split_masks = bundle.get("split_masks", {})
    train_mask = split_masks.get("east_west_train")
    if train_mask is None:
        train_mask = [True] * len(graphs)

    if args.sanitize:
        phase("sanitize_graph_list")
        n_fixed = sanitize_graph_list(graphs, show_progress=True)
        print(f"sanitize fixed={n_fixed}", flush=True)

    global_dim = 0
    if args.global_features_csv:
        phase(f"attach_global_features_from_csv {args.global_features_csv}")
        from gnn.global_graph_features import attach_global_features_from_csv

        stats = attach_global_features_from_csv(
            graphs, Path(args.global_features_csv), train_mask, verbose=True
        )
        global_dim = int(stats["global_feat_dim"])
        print(f"global_dim={global_dim} sample_gf={graphs[0].gf.shape}", flush=True)

    phase("DataLoader subset (first 8 train graphs)")
    train_idx = [i for i in range(len(graphs)) if train_mask[i]][:8]
    subset = [graphs[i] for i in train_idx]
    loader = DataLoader(subset, batch_size=min(args.batch_size, len(subset)), shuffle=False)

    if not args.forward:
        print("\nOK through load (+ optional sanitize/global). Re-run with --forward --device cuda", flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    phase(f"model.to({device}) + one batch forward")
    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=32,
        gat_heads=4,
        edge_dim=1,
        model_type="gat",
        global_dim=global_dim,
    ).to(device)
    batch = next(iter(loader)).to(device)
    print(
        f"batch: x={batch.x.shape} ei={batch.edge_index.shape} "
        f"y={batch.y.shape} gf={getattr(batch, 'gf', None)}",
        flush=True,
    )
    with torch.no_grad():
        out, _, _, _ = model(batch)
    print(f"forward ok pred={out.shape}", flush=True)
    del model, batch, out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("\nALL PHASES OK", flush=True)


if __name__ == "__main__":
    main()
