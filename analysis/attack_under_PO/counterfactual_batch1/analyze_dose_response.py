#!/usr/bin/env python3
"""
Post-process Batch-1 OAT results: dose-response curves + by-state / quadrant report.

Reads results_cf_batch1_2023_pooled/panel_scenario_summary.csv (or rebuilds from
unit summary.csv). Joins chronic/acute quadrants for differentiated policy text.

Framing (for manuscript):
  - Compare slopes of ΔL vs dose (%), not single-point ΔL at 20%.
  - Report per-state; link to four-quadrant classification.
  - No dollar cost conversion — "per unit change in intervention" only.

Usage:
  python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BATCH = ROOT / "results_cf_batch1_2023_pooled"
DEFAULT_QUADRANT = (
    ROOT
    / "results_mc_10km_panel_2018_2026"
    / "residual_analysis_2018_2023"
    / "chronic_acute_classification_CANONICAL.csv"
)
DOSE_PCTS = (0, 10, 20, 30, 50)

FAMILY_LABEL = {
    "CF-D": "NEVI corridor (main)",
    "CF-D-pop": "Population-weighted",
    "CF-D-null": "Uniform null (mean of 10 reps)",
    "CF-D-within": "Within-cluster (beyond resolution)",
    "CF-S": "Severity: N_aff −% (R_c×√(1−x))",
    "CF-T": "Duration reduction (−% T_i)",
}


def _parse_dose(row: pd.Series) -> float:
    if row.get("scenario") == "baseline" or row.get("family") == "baseline":
        return 0.0
    key = str(row["scenario"])
    for tok in key.replace("-", "_").split("_"):
        if tok.startswith("p") and tok[1:].isdigit():
            return float(tok[1:])
        if tok.startswith("m") and tok[1:].isdigit():
            return float(tok[1:])
    return float("nan")


def load_panel(batch_dir: Path) -> pd.DataFrame:
    panel_path = batch_dir / "panel_scenario_summary.csv"
    if panel_path.is_file():
        df = pd.read_csv(panel_path)
    else:
        parts = list(batch_dir.glob("*/summary.csv"))
        if not parts:
            raise SystemExit(f"No summaries under {batch_dir}")
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if "dose_pct" not in df.columns:
        df["dose_pct"] = df.apply(_parse_dose, axis=1)
    return df


def attach_quadrants(df: pd.DataFrame, quadrant_csv: Path) -> pd.DataFrame:
    if not quadrant_csv.is_file():
        df["quadrant"] = "unknown"
        return df
    q = pd.read_csv(quadrant_csv)
    # map unit / label / state
    key_cols = [c for c in ("state", "display_name") if c in q.columns]
    q = q[key_cols + ["quadrant", "quadrant_label", "n_nodes"]].copy()
    out = df.copy()
    out = out.merge(q, left_on="unit", right_on="state", how="left", suffixes=("", "_q"))
    miss = out["quadrant"].isna()
    if miss.any() and "label" in out.columns:
        out2 = df.loc[miss].drop(columns=[c for c in out.columns if c not in df.columns], errors="ignore")
        # retry on label
        m2 = df.merge(
            q,
            left_on="label",
            right_on="state",
            how="left",
        )
        out.loc[miss, "quadrant"] = m2.loc[miss, "quadrant"].values
        if "quadrant_label" in m2.columns:
            out.loc[miss, "quadrant_label"] = m2.loc[miss, "quadrant_label"].values
    out["quadrant"] = out["quadrant"].fillna("unknown")
    return out


def dose_response_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: unit × family × dose → ΔL and %ΔL.

    CF-D-null replicates are averaged at each (unit, dose) before ΔL.
    """
    work = df.copy()
    if (work["family"] == "CF-D-null").any():
        null_avg = (
            work[work["family"] == "CF-D-null"]
            .groupby(["unit", "family", "dose_pct"], as_index=False)["L_tilde_mean"]
            .mean()
        )
        work = pd.concat(
            [work[work["family"] != "CF-D-null"], null_avg],
            ignore_index=True,
        )

    base = work[work["family"] == "baseline"][["unit", "L_tilde_mean"]].rename(
        columns={"L_tilde_mean": "L0"}
    )
    fams = ("CF-D", "CF-D-pop", "CF-D-null", "CF-D-within", "CF-S", "CF-T")
    rows = []
    for fam in fams:
        for _, b in base.iterrows():
            rows.append(
                {
                    "unit": b["unit"],
                    "family": fam,
                    "dose_pct": 0.0,
                    "L_tilde_mean": b["L0"],
                    "delta_L_tilde": 0.0,
                    "delta_L_tilde_pct": 0.0,
                }
            )
    sub = work[work["family"].isin(fams)].copy()
    sub = sub.merge(base, on="unit", how="left")
    for _, r in sub.iterrows():
        L0 = float(r["L0"]) if pd.notna(r["L0"]) else float("nan")
        L = float(r["L_tilde_mean"])
        dL = L - L0 if np.isfinite(L0) else float("nan")
        dLp = 100.0 * dL / abs(L0) if np.isfinite(L0) and L0 != 0 else float("nan")
        rows.append(
            {
                "unit": r["unit"],
                "family": r["family"],
                "dose_pct": float(r["dose_pct"]),
                "L_tilde_mean": L,
                "delta_L_tilde": dL,
                "delta_L_tilde_pct": dLp,
            }
        )
    return pd.DataFrame(rows)


def fit_slopes(dose_df: pd.DataFrame) -> pd.DataFrame:
    """Per unit × family: OLS slope of ΔL_tilde_pct on dose_pct (per +1% intervention)."""
    rows = []
    for (unit, fam), g in dose_df.groupby(["unit", "family"]):
        g = g.sort_values("dose_pct")
        x = g["dose_pct"].astype(float).values
        y = g["delta_L_tilde_pct"].astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            slope = intercept = r = float("nan")
        else:
            slope, intercept, r, *_ = stats.linregress(x[mask], y[mask])
        rows.append(
            {
                "unit": unit,
                "family": fam,
                "slope_dLpct_per_dose_pct": float(slope),
                "intercept": float(intercept),
                "r_value": float(r),
                "n_doses": int(mask.sum()),
                "delta_at_20": float(g.loc[g["dose_pct"] == 20, "delta_L_tilde_pct"].mean())
                if (g["dose_pct"] == 20).any()
                else float("nan"),
                "delta_at_50": float(g.loc[g["dose_pct"] == 50, "delta_L_tilde_pct"].mean())
                if (g["dose_pct"] == 50).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_pooled_dose_response(dose_df: pd.DataFrame, out_path: Path) -> None:
    """Mean ± SE across states for CF-D (coverage expansion) vs CF-S."""
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor="white")
    colors = {
        "CF-D": "#1B4F72",
        "CF-D-pop": "#117A65",
        "CF-D-null": "#7F8C8D",
        "CF-S": "#B2182B",
        "CF-T": "#5DADE2",
    }
    styles = {"CF-D": "o-", "CF-D-pop": "s--", "CF-D-null": "^:", "CF-S": "o-"}
    for fam in ("CF-D", "CF-D-pop", "CF-D-null", "CF-S"):
        g = dose_df[dose_df["family"] == fam]
        if g.empty:
            continue
        agg = (
            g.groupby("dose_pct")["delta_L_tilde_pct"]
            .agg(["mean", "sem", "count"])
            .reset_index()
            .sort_values("dose_pct")
        )
        ax.errorbar(
            agg["dose_pct"],
            agg["mean"],
            yerr=agg["sem"],
            fmt=styles.get(fam, "o-"),
            color=colors.get(fam, "#333"),
            lw=1.4,
            markersize=5,
            capsize=3,
            label=FAMILY_LABEL.get(fam, fam),
        )
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_xlabel("Intervention intensity (%), one-at-a-time", fontsize=9)
    ax.set_ylabel(r"$\Delta L_s^{\mathrm{cum}}$ vs baseline (%)", fontsize=9)
    ax.set_xticks(list(DOSE_PCTS))
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_by_quadrant(dose_df: pd.DataFrame, slopes: pd.DataFrame, out_path: Path) -> None:
    """Facet dose-response mean by quadrant for CF-D vs CF-S."""
    if "quadrant" not in dose_df.columns:
        return
    quads = [q for q in dose_df["quadrant"].dropna().unique() if q != "unknown"]
    if not quads:
        return
    n = len(quads)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=True, facecolor="white")
    if n == 1:
        axes = [axes]
    colors = {"CF-D": "#1B4F72", "CF-D-pop": "#117A65", "CF-S": "#B2182B"}
    for ax, q in zip(axes, sorted(quads)):
        sub = dose_df[dose_df["quadrant"] == q]
        for fam in ("CF-D", "CF-D-pop", "CF-S"):
            g = sub[sub["family"] == fam]
            if g.empty:
                continue
            agg = (
                g.groupby("dose_pct", as_index=False)["delta_L_tilde_pct"]
                .mean()
                .sort_values("dose_pct")
            )
            ax.plot(
                agg["dose_pct"],
                agg["delta_L_tilde_pct"],
                "o-",
                color=colors[fam],
                lw=1.3,
                markersize=4,
                label=FAMILY_LABEL.get(fam, fam),
            )
        ax.axhline(0, color="#CCC", lw=0.8, ls="--")
        ax.set_title(str(q).replace("_", " "), fontsize=9)
        ax.set_xlabel("Dose %", fontsize=8)
        ax.set_xticks(list(DOSE_PCTS))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\Delta L$ (%)", fontsize=9)
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Dose-response by chronic/acute quadrant", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def policy_blurb(slopes: pd.DataFrame) -> str:
    """Short text for manuscript (no cost claims)."""
    # Compare CF-D vs CF-S slopes (more negative = larger vulnerability reduction)
    wide = slopes.pivot(index="unit", columns="family", values="slope_dLpct_per_dose_pct")
    if "CF-D" not in wide.columns or "CF-S" not in wide.columns:
        return "Insufficient families for CF-D vs CF-S slope comparison."
    d = wide["CF-D"]
    s = wide["CF-S"]
    # For reductions, slope should be negative; "S better" if S more negative
    s_better = (s < d).sum()
    d_better = (d < s).sum()
    n = int(wide.dropna(subset=["CF-D", "CF-S"]).shape[0])
    return (
        f"Across {n} units, OLS slopes of %ΔL on intervention intensity (%): "
        f"severity reduction steeper than NEVI corridor expansion in {s_better}/{n} units; "
        f"NEVI corridor steeper in {d_better}/{n}. "
        "Main CF-D places new hypernodes every 50 mi along interstate corridors, "
        "excluding sites <10 km from existing hypernodes (coverage expansion, not "
        "within-cluster densification). Population-weighted and uniform-null rules "
        "are robustness / null controls. Epicenters are locked to the baseline network "
        "(new nodes may be covered by R_c but never chosen as epicenter). "
        "Comparisons are per-unit intensity change, not dollar costs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OAT dose-response analysis for CF Batch 1")
    parser.add_argument("--batch-dir", type=str, default=str(DEFAULT_BATCH))
    parser.add_argument("--quadrant", type=str, default=str(DEFAULT_QUADRANT))
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir / "analysis_dose_response"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(batch_dir)
    panel = attach_quadrants(panel, Path(args.quadrant))
    dose = dose_response_table(panel)
    # attach quadrant onto dose table
    qmap = panel.drop_duplicates("unit")[["unit", "quadrant", "quadrant_label"]]
    dose = dose.merge(qmap, on="unit", how="left")
    slopes = fit_slopes(dose)
    slopes = slopes.merge(qmap, on="unit", how="left")

    dose.to_csv(out_dir / "dose_response_long.csv", index=False)
    slopes.to_csv(out_dir / "slopes_by_unit.csv", index=False)

    # quadrant summary of slopes
    if slopes["quadrant"].notna().any():
        qsum = (
            slopes[slopes["family"].isin(("CF-D", "CF-D-pop", "CF-D-null", "CF-S"))]
            .groupby(["quadrant", "family"])["slope_dLpct_per_dose_pct"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        qsum.to_csv(out_dir / "slopes_by_quadrant.csv", index=False)

    plot_pooled_dose_response(dose, out_dir / "figure_dose_response_pooled.png")
    plot_by_quadrant(dose, slopes, out_dir / "figure_dose_response_by_quadrant.png")

    blurb = policy_blurb(slopes)
    (out_dir / "policy_blurb.txt").write_text(blurb, encoding="utf-8")
    caption = (
        "Dose-response of cumulative vulnerability under one-at-a-time interventions "
        "(common-random-number Monte Carlo; 2023 network × pooled outages). "
        "Horizontal axis: intervention intensity (0–50%). "
        "Vertical axis: percent change in pooled $L_s^{\\mathrm{cum}}$ relative to baseline. "
        "Navy: NEVI corridor coverage expansion (main; new hypernodes every 50 mi on "
        "interstate, excluding sites <10 km from existing). Green: population-weighted "
        "placement. Gray: uniform spatial null (mean of 10 replicates). "
        "Red: severity as N_affected −% (impact radius scaled by √(1−x)). "
        "Epicenters are frozen from the baseline network; new nodes are disruption "
        "targets only. Within-cluster densification/capacity are beyond model resolution. "
        "Curves show cross-state means ± SEM. No dollar conversion."
    )
    (out_dir / "figure_dose_response_pooled_caption.txt").write_text(caption, encoding="utf-8")

    meta = {
        "dose_pcts": list(DOSE_PCTS),
        "framing": "OAT dose-response; compare slopes; no cost conversion",
        "n_units": int(dose["unit"].nunique()),
        "policy_blurb": blurb,
    }
    (out_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved under {out_dir}")
    print(blurb)


if __name__ == "__main__":
    main()
