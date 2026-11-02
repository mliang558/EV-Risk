#!/usr/bin/env python3
"""Export validated 2023 canonical results as manuscript-ready LaTeX.

The input directory must contain the compact canonical artifact:
``canonical_validation_summary.csv`` and
``canonical_rank_uncertainty.csv``.  The script cross-checks both files
before writing prose or tables so that an older ranking cannot be paired with
a newer simulation summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


UNIT_LABELS = {
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "Connecticut": "Connecticut--Rhode Island",
    "Dakotas": "North Dakota--South Dakota",
    "FL": "Florida",
    "GA": "Georgia",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "Maryland": "Maryland--Delaware--DC",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "SC": "South Carolina",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def unit_label(unit: str) -> str:
    return UNIT_LABELS.get(unit, unit)


def natural_list(items: list[str]) -> str:
    escaped = [latex_escape(unit_label(item)) for item in items]
    if len(escaped) == 1:
        return escaped[0]
    if len(escaped) == 2:
        return f"{escaped[0]} and {escaped[1]}"
    return ", ".join(escaped[:-1]) + f", and {escaped[-1]}"


def load_and_validate(artifact: Path, expected_units: int, expected_sims: int) -> pd.DataFrame:
    summary_path = artifact / "canonical_validation_summary.csv"
    ranks_path = artifact / "canonical_rank_uncertainty.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not ranks_path.is_file():
        raise FileNotFoundError(ranks_path)

    summary = pd.read_csv(summary_path)
    ranks = pd.read_csv(ranks_path)
    required_summary = {
        "unit",
        "n_sims",
        "W_mean",
        "L_tilde_mean",
        "L_event_mean",
    }
    required_ranks = {
        "metric",
        "unit",
        "mean",
        "mean_ci_low",
        "mean_ci_high",
        "rank",
        "rank_ci_low",
        "rank_ci_high",
        "prob_top5",
    }
    if not required_summary.issubset(summary.columns):
        raise ValueError(f"Summary is missing {sorted(required_summary - set(summary.columns))}")
    if not required_ranks.issubset(ranks.columns):
        raise ValueError(f"Rank file is missing {sorted(required_ranks - set(ranks.columns))}")
    if len(summary) != expected_units or summary["unit"].nunique() != expected_units:
        raise ValueError(f"Expected {expected_units} unique units in the summary")
    if not (summary["n_sims"] == expected_sims).all():
        raise ValueError(f"Every unit must contain {expected_sims} simulations")

    metric_frames: dict[str, pd.DataFrame] = {}
    for metric in ("L_cum", "L_event"):
        frame = ranks.loc[ranks["metric"] == metric].copy()
        if len(frame) != expected_units or frame["unit"].nunique() != expected_units:
            raise ValueError(f"Expected {expected_units} unique {metric} rank rows")
        if sorted(frame["rank"].astype(int)) != list(range(1, expected_units + 1)):
            raise ValueError(f"{metric} ranks are not a complete 1--{expected_units} sequence")
        metric_frames[metric] = frame

    cum = metric_frames["L_cum"].rename(
        columns={
            "mean": "L_cum",
            "mean_ci_low": "L_cum_ci_low",
            "mean_ci_high": "L_cum_ci_high",
            "rank": "rank_L_cum",
            "rank_ci_low": "rank_L_cum_ci_low",
            "rank_ci_high": "rank_L_cum_ci_high",
            "prob_top5": "prob_top5_L_cum",
        }
    )
    event = metric_frames["L_event"].rename(
        columns={
            "mean": "L_event",
            "rank": "rank_L_event",
        }
    )
    keep_cum = [
        "unit",
        "L_cum",
        "L_cum_ci_low",
        "L_cum_ci_high",
        "rank_L_cum",
        "rank_L_cum_ci_low",
        "rank_L_cum_ci_high",
        "prob_top5_L_cum",
    ]
    panel = summary.merge(cum[keep_cum], on="unit", validate="one_to_one")
    panel = panel.merge(
        event[["unit", "L_event", "rank_L_event"]],
        on="unit",
        validate="one_to_one",
    )
    if not np.allclose(panel["L_tilde_mean"], panel["L_cum"], rtol=1e-8, atol=1e-10):
        raise ValueError("L_cum means do not match canonical_validation_summary.csv")
    if not np.allclose(panel["L_event_mean"], panel["L_event"], rtol=1e-8, atol=1e-10):
        raise ValueError("L_event means do not match canonical_validation_summary.csv")
    numeric = panel.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Canonical result contains missing or non-finite values")

    panel["relative_ci_half_width"] = (
        (panel["L_cum_ci_high"] - panel["L_cum_ci_low"]) / 2 / panel["L_cum"]
    )
    panel["rank_gap"] = panel["rank_L_event"] - panel["rank_L_cum"]
    return panel.sort_values("rank_L_cum").reset_index(drop=True)


def write_main_table(panel: pd.DataFrame, path: Path) -> None:
    selected = pd.concat([panel.head(5), panel.tail(5)]).drop_duplicates("unit")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Highest- and lowest-ranked analytical units by 2023 node-normalized annual performance loss. Confidence intervals represent Monte Carlo uncertainty.}",
        r"\label{tab:canonical-spatial-ranking}",
        r"\begin{tabular}{rlccc}",
        r"\toprule",
        r"Rank & Analytical unit & $L_s^{\mathrm{cum}}$ & 95\% CI & Rank 95\% CI \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{int(row.rank_L_cum)} & {latex_escape(unit_label(row.unit))} & "
            f"{row.L_cum:.3f} & "
            f"[{row.L_cum_ci_low:.3f}, {row.L_cum_ci_high:.3f}] & "
            f"{int(row.rank_L_cum_ci_low)}--{int(row.rank_L_cum_ci_high)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_appendix_table(panel: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{longtable}{rlrrrr}",
        r"\caption{Complete 2023 canonical vulnerability ranking across 45 analytical units.}\label{tab:canonical-full-ranking}\\",
        r"\toprule",
        r"Rank & Analytical unit & $L_s^{\mathrm{cum}}$ & 95\% CI & Rank CI & $L_s^{\mathrm{event}}$ rank \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Rank & Analytical unit & $L_s^{\mathrm{cum}}$ & 95\% CI & Rank CI & $L_s^{\mathrm{event}}$ rank \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in panel.iterrows():
        lines.append(
            f"{int(row.rank_L_cum)} & {latex_escape(unit_label(row.unit))} & "
            f"{row.L_cum:.3f} & "
            f"[{row.L_cum_ci_low:.3f}, {row.L_cum_ci_high:.3f}] & "
            f"{int(row.rank_L_cum_ci_low)}--{int(row.rank_L_cum_ci_high)} & "
            f"{int(row.rank_L_event)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_text(panel: pd.DataFrame, path: Path) -> None:
    top = panel.head(5)
    bottom = panel.tail(5).sort_values("rank_L_cum", ascending=False)
    median_half_width = 100 * panel["relative_ci_half_width"].median()
    top_half_width = 100 * top["relative_ci_half_width"].median()
    rho, _ = spearmanr(panel["rank_L_cum"], panel["rank_L_event"])

    text = rf"""\subsection{{Spatial Patterns of Outage Vulnerability}}
\label{{sec:spatial-pattern}}

Figure~\ref{{fig:state-vulnerability-map}} shows the geographic distribution
of mean node-normalized annual performance loss, \Lcum{{}}, across the 45
analytical units in 2023. The five highest-ranked units were
{natural_list(top['unit'].tolist())}. The five lowest-ranked units were
{natural_list(bottom['unit'].tolist())}. These results demonstrate substantial
geographic variation in the expected duration-weighted loss of charging-network
efficiency under the common pooled 2018--2023 outage distribution.

The median 95\% Monte Carlo confidence-interval half-width was
{median_half_width:.1f}\% of the estimated mean across all units and
{top_half_width:.1f}\% among the five highest-ranked units. Confidence intervals
overlapped for several neighboring ranks, so the map is interpreted as showing
broad geographic differences rather than exact distinctions between adjacent
positions. Rankings based on mean per-event loss, \Levent{{}}, were positively
associated with the primary ranking (Spearman $\rho={rho:.2f}$), while rank
differences indicate that annual cumulative exposure and event-level structural
sensitivity capture related but distinct dimensions of outage vulnerability.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=45)
    parser.add_argument("--expected-sims", type=int, default=100)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_and_validate(artifact, args.expected_units, args.expected_sims)

    panel.to_csv(out_dir / "canonical_2023_spatial_results.csv", index=False)
    write_results_text(panel, out_dir / "spatial_pattern_results.tex")
    write_main_table(panel, out_dir / "canonical_2023_top_bottom.tex")
    write_appendix_table(panel, out_dir / "canonical_2023_full_ranking.tex")

    print("PASS: canonical summary and bootstrap ranks agree")
    print(f"PASS: {len(panel)} units x {args.expected_sims} simulations")
    print(f"Wrote LaTeX and CSV files under {out_dir}")
    print("\nTop five by L_cum:")
    print(panel.head(5)[["rank_L_cum", "unit", "L_cum"]].to_string(index=False))
    print("\nBottom five by L_cum:")
    print(panel.tail(5)[["rank_L_cum", "unit", "L_cum"]].to_string(index=False))


if __name__ == "__main__":
    main()
