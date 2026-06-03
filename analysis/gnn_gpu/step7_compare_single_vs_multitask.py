#!/usr/bin/env python3
"""
Compare multi-task (B+R joint) vs single-task (separate betweenness / random models).

Reads test_metrics.json from step7_evaluate_gnn.py or metrics embedded in checkpoints.

Usage (on step4_gpu):
  PYTHONPATH=. python -u step7_compare_single_vs_multitask.py \\
    --dataset pyg/pyg_dataset.pt \\
    --multitask pyg/train_gat_global_br/best_model.pt \\
    --betweenness pyg/train_gat_global_betweenness/best_model.pt \\
    --random pyg/train_gat_global_random/best_model.pt \\
    --out results_gnn/single_vs_multitask
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

from step7_build_comparison_table import eval_gnn_checkpoint, metrics_from_report_json


def head_r2_from_ckpt(ckpt_path: Path, dataset: Path, data_dir: Path, global_csv: Path | None, device: str) -> dict:
    met = eval_gnn_checkpoint(ckpt_path, dataset, data_dir, global_csv, device=device)
    return {
        "betweenness_r2": met.get("betweenness_r2"),
        "random_r2": met.get("random_r2"),
        "overall_r2": met.get("overall_r2"),
        "overall_mae": met.get("overall_mae"),
        "checkpoint": str(ckpt_path),
        "best_epoch": met.get("best_epoch"),
    }


def head_r2_from_eval_json(path: Path) -> dict:
    rep = json.loads(path.read_text(encoding="utf-8"))
    out = {"overall_r2": float(rep.get("overall", {}).get("r2", np.nan))}
    for head in ("betweenness", "random"):
        block = rep.get(head)
        if isinstance(block, dict) and "r2" in block:
            out[f"{head}_r2"] = float(block["r2"])
    tm = rep.get("test_metrics")
    if isinstance(tm, dict):
        rep = {**rep, **tm}
    if "test_betweenness_r2" in rep:
        out["betweenness_r2"] = float(rep["test_betweenness_r2"])
    if "test_random_r2" in rep:
        out["random_r2"] = float(rep["test_random_r2"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-task vs multi-task GNN comparison")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--global-csv", default="results_gnn/lgb_baseline/window_global_features.csv")
    parser.add_argument("--multitask", default="pyg/train_gat_global_br/best_model.pt")
    parser.add_argument("--betweenness", default="pyg/train_gat_global_betweenness/best_model.pt")
    parser.add_argument("--random", default="pyg/train_gat_global_random/best_model.pt")
    parser.add_argument(
        "--eval-multitask",
        default="results_gnn/eval_gat_global_br/test_metrics.json",
        help="If present, skip re-eval for multitask row",
    )
    parser.add_argument("--eval-betweenness", default=None)
    parser.add_argument("--eval-random", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results_gnn/single_vs_multitask")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()
    ds = Path(args.dataset).resolve()
    gcsv = Path(args.global_csv) if Path(args.global_csv).is_file() else None
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_row(name: str, ckpt: Path, eval_json: str | None) -> dict:
        row = {"setup": name}
        if eval_json:
            p = Path(eval_json)
            if not p.is_file():
                p = data_dir / eval_json
            if p.is_file():
                row.update(head_r2_from_eval_json(p))
                row["source"] = str(p)
                return row
        if ckpt.is_file():
            row.update(head_r2_from_ckpt(ckpt, ds, data_dir, gcsv, args.device))
            row["source"] = str(ckpt)
        else:
            row["source"] = "missing"
        return row

    mt = load_row("multi-task (B+R)", Path(args.multitask), args.eval_multitask)
    b = load_row("single-task betweenness", Path(args.betweenness), args.eval_betweenness)
    r = load_row("single-task random", Path(args.random), args.eval_random)

    # Stitched: use each single-task model on its own head
    stitched = {
        "setup": "stitched (B single + R single)",
        "betweenness_r2": b.get("betweenness_r2"),
        "random_r2": r.get("random_r2"),
        "overall_r2": np.nanmean(
            [b.get("betweenness_r2", np.nan), r.get("random_r2", np.nan)]
        ),
        "source": "betweenness_ckpt + random_ckpt",
    }

    rows = [mt, b, r, stitched]
    df = pd.DataFrame(rows)

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.3f}"

    lines = [
        "| Setup | Betweenness R² | Random R² | Overall R² |",
        "|-------|----------------|-----------|------------|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['setup']} | {fmt(row.get('betweenness_r2'))} | "
            f"{fmt(row.get('random_r2'))} | {fmt(row.get('overall_r2'))} |"
        )
    md = "\n".join(lines)
    print(md)
    df.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(md + "\n", encoding="utf-8")
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_dir}/comparison.md")


if __name__ == "__main__":
    main()
