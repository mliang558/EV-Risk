#!/usr/bin/env python3
"""
Plot a simple bar chart comparing California vs New York
using the MC node-normalized outage-attack metric.

Inputs (relative to Pro_directory):
- results/New York_attack_mc_node_normalized_posterior_lambda.csv
- results_CA/California_attack_mc_node_normalized_posterior_lambda.csv

Output:
- results/CA_NY_node_normalized_loss_bar.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # Paths to summary CSVs
    ny_path = (
        PROJECT_ROOT
        / "results"
        / "New York_attack_mc_node_normalized_posterior_lambda.csv"
    )
    ca_path = (
        PROJECT_ROOT
        / "results_CA"
        / "California_attack_mc_node_normalized_posterior_lambda.csv"
    )

    if not ny_path.exists():
        raise FileNotFoundError(f"Missing New York summary CSV: {ny_path}")
    if not ca_path.exists():
        raise FileNotFoundError(f"Missing California summary CSV: {ca_path}")

    ny = pd.read_csv(ny_path).iloc[0]
    ca = pd.read_csv(ca_path).iloc[0]

    states = ["California", "New York"]
    node_norm = [ca["node_normalized_loss"], ny["node_normalized_loss"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(states, node_norm, color=["tab:orange", "tab:blue"])

    ax.set_ylabel("node_normalized_loss\n(MC-averaged Annual Loss per node)")
    ax.set_title("California vs New York\nMC node-normalized outage vulnerability")

    # Annotate bar values
    for bar, val in zip(bars, node_norm):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out_path = PROJECT_ROOT / "results" / "CA_NY_node_normalized_loss_bar.png"
    PROJECT_ROOT.joinpath("results").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved CA vs NY bar plot to: {out_path}")


if __name__ == "__main__":
    main()

