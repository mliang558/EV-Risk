#!/usr/bin/env python3
"""
Aggregate enhancement experiments into one comparison table.

Fill experiments/enhancement_manifest.json or pass --manifest.

Example manifest entry:
  {"label": "GAT L5 ext", "eval_json": "results_gnn/eval_gat_L5_ext/test_metrics.json",
   "ckpt": "pyg/train_gat_global_L5_ext_br/best_model.pt", "notes": "5-layer + 18 global"}

Usage:
  PYTHONPATH=. python -u step7_build_enhancement_report.py \\
    --manifest experiments/enhancement_manifest.json \\
    --out results_gnn/enhancement_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step7_build_comparison_table import metrics_from_report_json


def load_row(entry: dict, data_dir: Path, dataset: Path, device: str) -> dict:
    label = entry.get("label", entry.get("name", "model"))
    row = {
        "model": label,
        "notes": entry.get("notes", ""),
        "betweenness_r2": np.nan,
        "random_r2": np.nan,
        "overall_r2": np.nan,
        "source": "",
    }
    eval_p = entry.get("eval_json") or entry.get("eval")
    if eval_p:
        p = Path(eval_p)
        if not p.is_file():
            p = data_dir / eval_p
        if p.is_file():
            met = metrics_from_report_json(p)
            if met:
                row.update(met)
                row["source"] = str(p)
                return row

    ckpt = entry.get("ckpt") or entry.get("checkpoint")
    if ckpt:
        from step7_build_comparison_table import eval_gnn_checkpoint

        gcsv = entry.get("global_csv")
        if gcsv:
            gcsv = Path(gcsv) if Path(gcsv).is_file() else data_dir / gcsv
        else:
            gcsv = None
            for cand in (
                data_dir / "results_gnn/global_extended/window_global_features.csv",
                data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
            ):
                if cand.is_file():
                    gcsv = cand
                    break
        p = Path(ckpt) if Path(ckpt).is_file() else data_dir / ckpt
        if p.is_file():
            met = eval_gnn_checkpoint(p, dataset, data_dir, gcsv, device=device)
            row.update({k: met.get(k, np.nan) for k in ("betweenness_r2", "random_r2", "overall_r2")})
            row["source"] = str(p)
    return row


def format_md(df: pd.DataFrame) -> str:
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.3f}"

    lines = [
        "| Model | Notes | Betweenness R² | Random R² | Overall R² |",
        "|-------|-------|----------------|-----------|------------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r.get('notes', '')} | "
            f"{fmt(r.get('betweenness_r2'))} | {fmt(r.get('random_r2'))} | {fmt(r.get('overall_r2'))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--manifest", default="experiments/enhancement_manifest.json")
    parser.add_argument("--out", default="results_gnn/enhancement_report")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        manifest_path = data_dir / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(entries, dict):
        entries = entries.get("experiments", entries.get("runs", []))

    rows = [load_row(e, data_dir, Path(args.dataset).resolve(), args.device) for e in entries]
    df = pd.DataFrame(rows)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    md = format_md(df)
    print(md)
    df.to_csv(out_dir / "enhancement_table.csv", index=False)
    (out_dir / "enhancement_table.md").write_text(md + "\n", encoding="utf-8")
    with open(out_dir / "enhancement_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_dir}/enhancement_table.md")


if __name__ == "__main__":
    main()
