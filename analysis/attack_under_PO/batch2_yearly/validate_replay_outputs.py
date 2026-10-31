#!/usr/bin/env python3
"""Validate fixed-2023-network historical replay outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = (2018, 2019, 2020, 2021, 2022, 2023)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sims", type=int, default=100)
    parser.add_argument("--expected-units", type=int, default=45)
    args = parser.parse_args()

    manifest_path = args.out / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("families") != ["B"]:
        raise ValueError("Manifest is not a B-only run")
    if manifest.get("event_mode") != "replay":
        raise ValueError("Manifest does not certify replay mode")
    if sorted(manifest.get("years", [])) != list(YEARS):
        raise ValueError("Manifest does not contain outage years 2018--2023")
    if int(manifest.get("n_sims", -1)) != args.expected_sims:
        raise ValueError("Manifest simulation count is incorrect")

    summaries = sorted((args.out / "B").glob("net2023_out*/*/summary.csv"))
    expected = args.expected_units * len(YEARS)
    if len(summaries) != expected:
        raise ValueError(f"Expected {expected} summaries, found {len(summaries)}")

    observed_pairs: set[tuple[str, int]] = set()
    for summary_path in summaries:
        summary = pd.read_csv(summary_path).iloc[0]
        unit = str(summary["unit"])
        year = int(summary["year_outage"])
        observed_pairs.add((unit, year))

        if str(summary["family"]) != "B" or int(summary["year_network"]) != 2023:
            raise ValueError(f"Wrong scenario identity: {summary_path}")
        if str(summary["event_mode"]) != "replay":
            raise ValueError(f"Non-replay output: {summary_path}")
        if int(summary["n_sims"]) != args.expected_sims:
            raise ValueError(f"Wrong simulation count: {summary_path}")
        if any(int(summary[name]) != 0 for name in (
            "n_fb_county", "n_fb_state_year", "n_fb_state_pooled"
        )):
            raise ValueError(f"Replay used a severity fallback: {summary_path}")

        metrics_path = summary_path.with_name("metrics_baseline.csv")
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != args.expected_sims:
            raise ValueError(f"Wrong metrics row count: {metrics_path}")
        numeric = metrics[
            ["n_events", "n_nodes", "baseline_efficiency", "L_cum",
             "L_event", "total_rel_loss_x_duration", "P_hit"]
        ].astype(float)
        if not np.isfinite(numeric).all().all():
            raise ValueError(f"Non-finite replay metric: {metrics_path}")
        if metrics["n_events"].nunique() != 1 or int(metrics["n_events"].iloc[0]) <= 0:
            raise ValueError(f"Observed-event count changed across replications: {metrics_path}")
        if not metrics["P_hit"].between(0, 1).all():
            raise ValueError(f"Invalid hit probability: {metrics_path}")
        if not np.allclose(
            metrics["L_cum"] * metrics["n_nodes"],
            metrics["total_rel_loss_x_duration"],
        ):
            raise ValueError(f"Cumulative-loss identity failed: {metrics_path}")
        if not np.allclose(
            metrics["L_event"] * metrics["n_events"],
            metrics["total_rel_loss_x_duration"],
        ):
            raise ValueError(f"Per-event-loss identity failed: {metrics_path}")

    for year in YEARS:
        count = sum(pair_year == year for _, pair_year in observed_pairs)
        if count != args.expected_units:
            raise ValueError(f"Outage year {year} has {count} units, expected {args.expected_units}")

    print(
        f"PASS: replay B has {expected} unit-years, fixed 2023 networks, "
        f"{args.expected_sims} replications, no severity fallbacks, and valid losses"
    )


if __name__ == "__main__":
    main()
