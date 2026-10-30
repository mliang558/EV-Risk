#!/usr/bin/env python3
"""Summarize intervention dose-response results and draw the main figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[3]
DOSES = (0, 10, 20, 30, 50)
FAMILIES = ("CF-D", "CF-S")
LABELS = {"CF-D": "Corridor expansion", "CF-S": "Severity reduction"}
COLORS = {"CF-D": "#1B4F72", "CF-S": "#B2182B"}


def _panel(batch_dir: Path) -> pd.DataFrame:
    path = batch_dir / "panel_scenario_summary.csv"
    if path.is_file():
        return pd.read_csv(path)
    parts = sorted(batch_dir.glob("*/summary.csv"))
    if not parts:
        raise FileNotFoundError(f"No unit summaries under {batch_dir}")
    return pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)


def dose_response(panel: pd.DataFrame) -> pd.DataFrame:
    baseline = panel[panel["family"] == "baseline"].set_index("unit")["L_tilde_mean"]
    rows = []
    for row in panel[panel["family"].isin(FAMILIES)].to_dict("records"):
        loss0 = float(baseline[row["unit"]])
        loss = float(row["L_tilde_mean"])
        rows.append(
            {
                "unit": row["unit"],
                "family": row["family"],
                "dose_pct": float(row["dose_pct"]),
                "loss": loss,
                "change_pct": 100 * (loss - loss0) / abs(loss0),
            }
        )
    result = pd.DataFrame(rows).sort_values(["unit", "family", "dose_pct"])
    counts = result.groupby(["unit", "family"])["dose_pct"].nunique()
    bad = counts[counts != len(DOSES)]
    if not bad.empty:
        raise ValueError(f"Each unit-family needs doses {DOSES}:\n{bad}")
    return result


def slopes_by_unit(dose: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (unit, family), group in dose.groupby(["unit", "family"]):
        fit = linregress(group["dose_pct"], group["change_pct"])
        rows.append(
            {
                "unit": unit,
                "family": family,
                "slope_per_percentage_point": fit.slope,
                "r_value": fit.rvalue,
                "change_at_20_pct": group.loc[group["dose_pct"] == 20, "change_pct"].iloc[0],
                "change_at_50_pct": group.loc[group["dose_pct"] == 50, "change_pct"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def plot(dose: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    for family in FAMILIES:
        summary = (
            dose[dose["family"] == family]
            .groupby("dose_pct")["change_pct"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        axis.errorbar(
            summary["dose_pct"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            capsize=3,
            color=COLORS[family],
            label=LABELS[family],
        )
    axis.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    axis.set(xlabel="Intervention intensity (%)", ylabel="Change in node-normalized loss (%)")
    axis.set_xticks(DOSES)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    batch = args.batch_dir if args.batch_dir.is_absolute() else ROOT / args.batch_dir
    out = args.out_dir or batch / "analysis_dose_response"
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    dose = dose_response(_panel(batch))
    slopes = slopes_by_unit(dose)
    dose.to_csv(out / "dose_response_long.csv", index=False)
    slopes.to_csv(out / "slopes_by_unit.csv", index=False)
    plot(dose, out / "figure_dose_response.png")

    comparison = slopes.pivot(index="unit", columns="family", values="slope_per_percentage_point")
    n_severity = int((comparison["CF-S"] < comparison["CF-D"]).sum())
    print(
        f"Saved {out}. Severity reduction has the steeper loss reduction in "
        f"{n_severity}/{len(comparison)} analytical units."
    )


if __name__ == "__main__":
    main()
