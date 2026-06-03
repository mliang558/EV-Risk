#!/usr/bin/env python3
"""Pack data/step4_gpu/subgraphs/*.npz into one archive for upload."""

from __future__ import annotations

import argparse
from pathlib import Path

from step4_archive_utils import pack_subgraphs_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Step-4 subgraph npz files")
    parser.add_argument("--data-dir", default="data/step4_gpu")
    parser.add_argument("--format", choices=("zip", "tar.gz"), default="zip")
    parser.add_argument("--out", default=None, help="Output archive path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    subgraphs_dir = data_dir / "subgraphs"
    if not subgraphs_dir.is_dir():
        raise FileNotFoundError(f"Missing {subgraphs_dir}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
    else:
        out = data_dir / ("subgraphs.zip" if args.format == "zip" else "subgraphs.tar.gz")

    pack_subgraphs_dir(subgraphs_dir, out, fmt=args.format)
    mb = out.stat().st_size / (1024 * 1024)
    n = len(list(subgraphs_dir.glob("*.npz")))
    print(f"Packed {n} files -> {out} ({mb:.1f} MB)")
