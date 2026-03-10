#!/usr/bin/env python3
"""
Plot charging network results: (1) network characteristics by state,
(2) performance loss vs removal_pct by strategy,
(3) per state: efficiency loss vs removal_pct, one curve per year.

Usage:
  python analysis/plot_charging_network_results.py --run-dir "outputs/charging_network/run__..."
  python analysis/plot_charging_network_results.py --by-year --base-dir outputs/charging_network --out-dir outputs/charging_network/plots_by_year
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_all_csv(path: Path, pattern: str):
    """Find first CSV matching pattern in path (e.g. network_characteristics_ALL_*.csv)."""
    files = list(path.glob(pattern))
    if not files:
        return None
    return files[0]


def plot_network_chars(run_dir: Path, out_dir: Path):
    """Bar charts: global_efficiency and num_nodes by state."""
    networks_dir = run_dir / "networks"
    f = find_all_csv(networks_dir, "network_characteristics_ALL_*.csv")
    if f is None:
        print(f"No network_characteristics_ALL_*.csv in {networks_dir}")
        return

    df = pd.read_csv(f)
    df = df.sort_values("state")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Global efficiency by state
    ax = axes[0]
    ax.bar(range(len(df)), df["global_efficiency"], color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["state"], rotation=90, fontsize=8)
    ax.set_ylabel("Global efficiency")
    ax.set_title("Network: global efficiency by state")
    ax.grid(axis="y", alpha=0.3)

    # Num nodes by state
    ax = axes[1]
    ax.bar(range(len(df)), df["num_nodes"], color="coral", alpha=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["state"], rotation=90, fontsize=8)
    ax.set_ylabel("Number of nodes")
    ax.set_xlabel("State")
    ax.set_title("Network: number of nodes by state")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "network_chars_by_state.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'network_chars_by_state.png'}")


def plot_attack_loss_by_strategy(run_dir: Path, out_dir: Path, state: str = None):
    """Line plot: efficiency_loss_pct vs removal_pct, one line per strategy."""
    attacks_dir = run_dir / "attacks"
    f = find_all_csv(attacks_dir, "attack_results_ALL_*.csv")
    if f is None:
        print(f"No attack_results_ALL_*.csv in {attacks_dir}")
        return

    df = pd.read_csv(f)
    if state:
        df = df[df["state"] == state].copy()

    strategies = df["strategy"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in strategies:
        sub = df[df["strategy"] == s].groupby("removal_pct", as_index=False).agg(
            mean_loss=("efficiency_loss_pct", "mean"),
            std_loss=("efficiency_loss_pct", "std"),
        )
        sub = sub.sort_values("removal_pct")
        ax.plot(
            sub["removal_pct"] * 100,
            sub["mean_loss"],
            label=s,
            marker="o",
            markersize=4,
        )
        if sub["std_loss"].notna().any():
            ax.fill_between(
                sub["removal_pct"] * 100,
                sub["mean_loss"] - sub["std_loss"].fillna(0),
                sub["mean_loss"] + sub["std_loss"].fillna(0),
                alpha=0.2,
            )

    ax.set_xlabel("Node removal (%)")
    ax.set_ylabel("Efficiency loss (%)")
    title = f"Performance loss vs removal fraction"
    if state:
        title += f" — {state}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{state}" if state else "_all_states"
    fig.savefig(out_dir / f"attack_loss_by_strategy{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / f'attack_loss_by_strategy{suffix}.png'}")


def plot_attack_loss_by_state(run_dir: Path, out_dir: Path, strategy: str = "random"):
    """Bar or line: for one strategy, efficiency_loss at fixed removal_pct by state."""
    attacks_dir = run_dir / "attacks"
    f = find_all_csv(attacks_dir, "attack_results_ALL_*.csv")
    if f is None:
        return

    df = pd.read_csv(f)
    df = df[df["strategy"] == strategy]
    # Pick a few removal levels to show
    for removal in [0.2, 0.5]:
        sub = df[df["removal_pct"] == removal].groupby("state", as_index=False)["efficiency_loss_pct"].mean()
        sub = sub.sort_values("efficiency_loss_pct", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sub["state"], sub["efficiency_loss_pct"], color="teal", alpha=0.8)
        ax.set_xlabel("Efficiency loss (%)")
        ax.set_ylabel("State")
        ax.set_title(f"Performance loss at {int(removal*100)}% removal — strategy: {strategy}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"attack_loss_by_state_{strategy}_pct{int(removal*100)}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved attack_loss_by_state_{strategy}_pct{int(removal*100)}.png")


def _year_from_run_dir(name: str) -> int | None:
    """Extract year from dir name like 'run__alt_fuel_stations_historical_day_(Jan_1_2018)' -> 2018."""
    m = re.search(r"\(Jan_1_(\d{4})\)", name)
    return int(m.group(1)) if m else None


def plot_loss_by_year_per_state(base_dir: Path, out_dir: Path, strategy: str = "random"):
    """
    For each state: one figure. X = node removal (%), Y = efficiency loss (%).
    One curve per year (from different run__* directories under base_dir).
    """
    base_dir = Path(base_dir)
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run__")])
    year_to_dir = {}
    for d in run_dirs:
        y = _year_from_run_dir(d.name)
        if y is not None:
            year_to_dir[y] = d

    if not year_to_dir:
        print(f"No run__* directories with (Jan_1_YYYY) found under {base_dir}")
        return

    # Load attack results per year
    series_by_state_year = {}  # (state, year) -> (removal_pct array, loss array)
    states = set()
    for year in sorted(year_to_dir.keys()):
        run_dir = year_to_dir[year]
        attacks_dir = run_dir / "attacks"
        f = find_all_csv(attacks_dir, "attack_results_ALL_*.csv")
        if f is None:
            continue
        df = pd.read_csv(f)
        df = df[df["strategy"] == strategy]
        for state in df["state"].unique():
            states.add(state)
            sub = df[df["state"] == state].groupby("removal_pct", as_index=False)["efficiency_loss_pct"].mean()
            sub = sub.sort_values("removal_pct")
            series_by_state_year[(state, year)] = (
                sub["removal_pct"].values * 100,
                sub["efficiency_loss_pct"].values,
            )

    # Colors for years (consistent across states)
    years_sorted = sorted(year_to_dir.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(years_sorted)))
    year_color = {y: colors[i] for i, y in enumerate(years_sorted)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in sorted(states):
        fig, ax = plt.subplots(figsize=(8, 5))
        for year in years_sorted:
            key = (state, year)
            if key not in series_by_state_year:
                continue
            x, y = series_by_state_year[key]
            ax.plot(x, y, label=str(year), color=year_color[year], marker="o", markersize=3)
        ax.set_xlabel("Node removal (%)")
        ax.set_ylabel("Efficiency loss (%)")
        ax.set_title(f"{state}: performance loss vs removal — strategy = {strategy}")
        ax.legend(title="Year")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 90)
        fig.savefig(out_dir / f"attack_loss_vs_removal_{state}_{strategy}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved attack_loss_vs_removal_{state}_{strategy}.png")
    print(f"Done. Plots in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot charging network and attack results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="outputs/charging_network/run__alt_fuel_stations_historical_day_(Jan_1_2018)",
        help="Path to one run output (e.g. run__..._(Jan_1_2018))",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to save plots (default: <run_dir>/plots)",
    )
    parser.add_argument("--state", type=str, default=None, help="Plot attack loss for this state only (e.g. FL)")
    parser.add_argument(
        "--by-year",
        action="store_true",
        help="Plot per state: X=removal%%, Y=efficiency loss%%, one curve per year (requires multiple run__* dirs)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/charging_network",
        help="Base dir containing run__* subdirs (used with --by-year)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Attack strategy for by-year plot (default: random)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.by_year:
        base_dir = project_root / args.base_dir
        out_dir = project_root / (args.out_dir or str(base_dir / "plots_by_year"))
        if not base_dir.exists():
            print(f"Base directory not found: {base_dir}")
            return
        print(f"Base dir: {base_dir}")
        print(f"Plot output: {out_dir}")
        plot_loss_by_year_per_state(base_dir, out_dir, strategy=args.strategy)
        print("Done.")
        return

    run_dir = project_root / args.run_dir
    out_dir = project_root / (args.out_dir or str(run_dir / "plots"))

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    print(f"Run dir: {run_dir}")
    print(f"Plot output: {out_dir}")

    plot_network_chars(run_dir, out_dir)
    plot_attack_loss_by_strategy(run_dir, out_dir, state=args.state)
    plot_attack_loss_by_state(run_dir, out_dir, strategy="random")
    plot_attack_loss_by_state(run_dir, out_dir, strategy="degree")

    print("Done.")


if __name__ == "__main__":
    main()
