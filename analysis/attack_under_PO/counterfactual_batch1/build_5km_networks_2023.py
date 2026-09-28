#!/usr/bin/env python3
"""
Build 2023 5 km hypernode networks for scenario K.

Writes to a SEPARATE root (does not touch 10 km graphs):
  outputs/network_graph_5km_2018_2026/2023/...

Usage:
  python analysis/attack_under_PO/counterfactual_batch1/build_5km_networks_2023.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis"
for p in (ROOT, ANALYSIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_panel = ANALYSIS / "build_10km_panel_2018_2026.py"
if not _panel.is_file():
    raise SystemExit(
        f"Missing {_panel}\n"
        "Needed only for scenario K. Either:\n"
        "  curl -fsSL -o analysis/build_10km_panel_2018_2026.py \\\n"
        "    https://raw.githubusercontent.com/mliang558/EV-Risk/main/analysis/build_10km_panel_2018_2026.py\n"
        "or drop K from FAMILIES (default is baseline,CF-D,CF-S,U)."
    )

import build_10km_panel_2018_2026 as b10  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 5km networks for K scenario")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "network_graph_5km_2018_2026"),
    )
    parser.add_argument("--radius-m", type=float, default=5000.0)
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(ROOT / "data" / "processed"),
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    processed = Path(args.processed_dir)
    if not processed.is_absolute():
        processed = ROOT / processed

    # Temporarily override module constant used by build_year
    old = b10.CLUSTER_RADIUS_M
    b10.CLUSTER_RADIUS_M = float(args.radius_m)
    try:
        summary = b10.build_year(int(args.year), out_root, processed, verbose=True)
    finally:
        b10.CLUSTER_RADIUS_M = old

    print(f"Done: {len(summary)} units @ {args.radius_m:.0f} m -> {out_root / str(args.year)}")


if __name__ == "__main__":
    main()
