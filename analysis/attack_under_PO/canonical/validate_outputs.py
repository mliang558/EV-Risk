#!/usr/bin/env python3
"""Audit canonical outputs and metric identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def close(a: pd.Series, b: pd.Series, *, atol: float = 1e-10) -> bool:
    return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), atol=atol, rtol=1e-8, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-sims", type=int, default=100)
    parser.add_argument("--expected-units", type=int, default=45)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    manifest_path = out / "batch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scenarios") != ["baseline"]:
        raise ValueError(f"Canonical output must contain baseline only: {manifest.get('scenarios')}")
    if manifest.get("state_rate_definition") != "Lambda_s = sum_c lambda_c":
        raise ValueError("Manifest does not certify the state sum-rate definition")

    unit_dirs = sorted(p.parent for p in out.glob("*/metrics_baseline.csv"))
    if len(unit_dirs) != args.expected_units:
        raise ValueError(f"Expected {args.expected_units} completed units, found {len(unit_dirs)}")

    audit_rows: list[dict] = []
    for unit_dir in unit_dirs:
        metrics = pd.read_csv(unit_dir / "metrics_baseline.csv")
        if len(metrics) != args.expected_sims:
            raise ValueError(f"{unit_dir.name}: expected {args.expected_sims} sims, found {len(metrics)}")
        required = {
            "n_events", "n_nodes", "n_hit_events", "P_hit", "L_event",
            "E_loss_given_hit", "L_tilde", "total_rel_loss_x_duration",
            "total_rel_loss_x_duration_hit", "n_fallback_events",
        }
        if not required.issubset(metrics.columns):
            raise ValueError(f"{unit_dir.name}: missing metrics {sorted(required - set(metrics.columns))}")
        if (metrics["n_events"] <= 0).any() or (metrics["n_nodes"] < 2).any():
            raise ValueError(f"{unit_dir.name}: invalid event or node count")
        if not close(metrics["L_tilde"], metrics["total_rel_loss_x_duration"] / metrics["n_nodes"]):
            raise AssertionError(f"{unit_dir.name}: L_tilde identity failed")
        if not close(metrics["P_hit"], metrics["n_hit_events"] / metrics["n_events"]):
            raise AssertionError(f"{unit_dir.name}: P_hit identity failed")
        if not close(metrics["L_event"], metrics["total_rel_loss_x_duration"] / metrics["n_events"]):
            raise AssertionError(f"{unit_dir.name}: L_event identity failed")
        hit = metrics["n_hit_events"] > 0
        if hit.any() and not close(
            metrics.loc[hit, "E_loss_given_hit"],
            metrics.loc[hit, "total_rel_loss_x_duration_hit"] / metrics.loc[hit, "n_hit_events"],
        ):
            raise AssertionError(f"{unit_dir.name}: conditional-loss identity failed")
        if ((metrics["P_hit"] < 0) | (metrics["P_hit"] > 1)).any():
            raise ValueError(f"{unit_dir.name}: P_hit outside [0,1]")
        audit_rows.append(
            {
                "unit": unit_dir.name,
                "n_sims": len(metrics),
                "n_events_mean": metrics["n_events"].mean(),
                "W_mean": metrics["total_rel_loss_x_duration"].mean(),
                "L_tilde_mean": metrics["L_tilde"].mean(),
                "P_hit_mean": metrics["P_hit"].mean(),
                "L_event_mean": metrics["L_event"].mean(),
                "E_loss_given_hit_mean": metrics["E_loss_given_hit"].mean(),
                "fallback_event_share": (
                    metrics["n_fallback_events"].sum() / metrics["n_events"].sum()
                ),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("unit")
    audit.to_csv(out / "canonical_validation_summary.csv", index=False)
    print(audit.to_string(index=False))
    print("PASS: canonical coverage, loss identities, and probability bounds are consistent")


if __name__ == "__main__":
    main()
