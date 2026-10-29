#!/usr/bin/env python3
"""Monte Carlo uncertainty and rank stability for canonical outputs.

The same simulation IDs are resampled for every analytical unit. This paired
bootstrap preserves the common-random-number design used by the simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "L_cum": "L_tilde",
    "L_event": "L_event",
}


def load_metric_panel(
    out: Path,
    column: str,
    expected_units: int,
    expected_sims: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    paths = sorted(out.glob("*/metrics_baseline.csv"))
    if len(paths) != expected_units:
        raise ValueError(
            f"Expected {expected_units} unit files, found {len(paths)} in {out}"
        )

    units: list[str] = []
    series: list[pd.Series] = []
    reference_ids: np.ndarray | None = None

    for path in paths:
        frame = pd.read_csv(path)
        required = {"sim_id", column}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing {sorted(required - set(frame.columns))}")
        if frame["sim_id"].duplicated().any():
            raise ValueError(f"{path}: duplicate sim_id values")

        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"{path}: {column} contains missing or non-finite values")

        item = pd.Series(values.to_numpy(float), index=frame["sim_id"].to_numpy())
        item = item.sort_index()
        sim_ids = item.index.to_numpy()
        if len(item) != expected_sims:
            raise ValueError(
                f"{path.parent.name}: expected {expected_sims} simulations, found {len(item)}"
            )
        if reference_ids is None:
            reference_ids = sim_ids
        elif not np.array_equal(reference_ids, sim_ids):
            raise ValueError(f"{path.parent.name}: sim_id values do not match other units")

        units.append(path.parent.name)
        series.append(item)

    assert reference_ids is not None
    panel = np.column_stack([item.to_numpy(float) for item in series])
    return units, reference_ids, panel


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based descending ranks for a vector or row-wise matrix."""
    if values.ndim == 1:
        order = np.argsort(-values, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(values) + 1)
        return ranks
    order = np.argsort(-values, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    target = np.broadcast_to(np.arange(1, values.shape[1] + 1), order.shape)
    np.put_along_axis(ranks, order, target, axis=1)
    return ranks


def bootstrap_summary(
    units: list[str],
    panel: np.ndarray,
    metric_name: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_sims, n_units = panel.shape
    observed_mean = panel.mean(axis=0)
    observed_rank = descending_ranks(observed_mean)
    mc_se = panel.std(axis=0, ddof=1) / np.sqrt(n_sims)

    sampled = rng.integers(0, n_sims, size=(n_bootstrap, n_sims))
    bootstrap_means = panel[sampled].mean(axis=1)
    bootstrap_ranks = descending_ranks(bootstrap_means)

    mean_low, mean_high = np.quantile(
        bootstrap_means, [0.025, 0.975], axis=0
    )
    rank_q_low, rank_q_high = np.quantile(
        bootstrap_ranks, [0.025, 0.975], axis=0
    )

    result = pd.DataFrame(
        {
            "metric": metric_name,
            "unit": units,
            "mean": observed_mean,
            "mc_se": mc_se,
            "mean_ci_low": mean_low,
            "mean_ci_high": mean_high,
            "rank": observed_rank,
            "rank_median": np.median(bootstrap_ranks, axis=0),
            "rank_ci_low": np.floor(rank_q_low).astype(int),
            "rank_ci_high": np.ceil(rank_q_high).astype(int),
            "prob_top5": (bootstrap_ranks <= 5).mean(axis=0),
            "prob_top10": (bootstrap_ranks <= 10).mean(axis=0),
        }
    )
    return result.sort_values("rank").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-bootstrap confidence intervals and rank stability"
    )
    parser.add_argument("--out", required=True, help="Canonical output directory")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--expected-units", type=int, default=45)
    parser.add_argument("--expected-sims", type=int, default=100)
    args = parser.parse_args()

    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    out = Path(args.out).resolve()
    rng = np.random.default_rng(args.seed)
    tables: list[pd.DataFrame] = []

    for metric_name, column in METRICS.items():
        units, sim_ids, panel = load_metric_panel(
            out, column, args.expected_units, args.expected_sims
        )
        table = bootstrap_summary(
            units, panel, metric_name, args.bootstrap, rng
        )
        tables.append(table)
        print(f"\n=== {metric_name} rank stability ===")
        print(
            table[
                [
                    "rank",
                    "unit",
                    "mean",
                    "mean_ci_low",
                    "mean_ci_high",
                    "rank_ci_low",
                    "rank_ci_high",
                    "prob_top5",
                ]
            ].to_string(index=False)
        )

    combined = pd.concat(tables, ignore_index=True)
    output_csv = out / "canonical_rank_uncertainty.csv"
    combined.to_csv(output_csv, index=False)

    metadata = {
        "method": "paired nonparametric bootstrap over aligned Monte Carlo sim_id values",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "n_units": args.expected_units,
        "n_simulations": args.expected_sims,
        "metrics": METRICS,
        "output": output_csv.name,
    }
    (out / "canonical_rank_uncertainty_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    main()
