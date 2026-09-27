#!/usr/bin/env python3
"""
Section 4.3 (MAIN) — Exposure × structure decomposition of event-mean loss:

    L_event = P(hit) × E[loss | hit]

- P(hit): share of outage events that disrupt ≥1 hypernode (spatial overlap /
  exposure of the charging network to the outage footprint).
- E[loss | hit]: mean relative efficiency loss × duration among hit events
  (conditional structural impact).

This answers "how much is exposure vs structure" more cleanly than regressions
where severity enters both sides. Panel / cluster regressions (former 4.3.2)
are demoted to supplementary.

Zero-loss events are expected under population-weighted epicenters (most real
outages miss EV stations). Existing MC outputs already record n_zero_loss_events;
this script reconstructs the decomposition from per-sim metrics when explicit
P_hit / E_loss_given_hit columns are absent.

Usage:
  python analysis/attack_under_PO/section_4_3_exposure_structure.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_mc_10km_panel_2018_2026"
YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
OUT = RESULTS / "residual_analysis_2018_2023"
QUADRANT = OUT / "chronic_acute_classification_CANONICAL.csv"
POOLED = OUT / "state_pooled_residual_ranking.csv"
EPS = 1e-15


def _load_n_nodes_map() -> dict[str, float]:
    """Best-effort |V| per state from panel or yearly files."""
    panel = RESULTS / "panel_mc_metrics.csv"
    if panel.is_file():
        df = pd.read_csv(panel)
        if "n_nodes" in df.columns and "state" in df.columns:
            return (
                df.groupby("state")["n_nodes"]
                .mean()
                .astype(float)
                .to_dict()
            )
    return {}


def load_decomposition_from_per_sim(
    results_dir: Path,
    years: tuple[int, ...] = YEARS,
) -> pd.DataFrame:
    """
    From <year>/<state>_mc_per_sim_metrics.csv:

      P_hit_sim = 1 - n_zero / n_events
      L_event_sim = L_tilde * n_nodes / n_events
                  = P_hit × E[loss|hit]

    When P_hit / E_loss_given_hit columns exist (new MC runs), use them directly.
    """
    n_nodes_map = _load_n_nodes_map()
    rows: list[dict] = []

    for year in years:
        year_dir = results_dir / str(year)
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*_mc_per_sim_metrics.csv")):
            state = path.name.replace("_mc_per_sim_metrics.csv", "")
            sim = pd.read_csv(path)
            if sim.empty or "n_events" not in sim.columns:
                continue

            n_nodes = float(n_nodes_map.get(state, np.nan))
            if not np.isfinite(n_nodes) or n_nodes <= 0:
                # try yearly file
                yr_path = year_dir / f"{state}_attack_yearly_mc_posterior_lambda.csv"
                if yr_path.is_file():
                    yr = pd.read_csv(yr_path)
                    if "n_nodes" in yr.columns and len(yr):
                        n_nodes = float(yr["n_nodes"].iloc[0])

            if "P_hit" in sim.columns and "E_loss_given_hit" in sim.columns:
                p_hit = sim["P_hit"].astype(float)
                e_hit = sim["E_loss_given_hit"].astype(float)
                if "L_event" in sim.columns:
                    l_ev = sim["L_event"].astype(float)
                else:
                    l_ev = p_hit * e_hit
            else:
                n_ev = sim["n_events"].astype(float).clip(lower=1)
                n_zero = sim["n_zero_loss_events"].astype(float)
                n_hit = (n_ev - n_zero).clip(lower=0)
                p_hit = n_hit / n_ev
                # L_tilde = total_rel / |V|  ⇒  total_rel = L_tilde * |V|
                # L_event = total_rel / n_ev
                l_tilde = sim["L_tilde"].astype(float)
                total_rel = l_tilde * n_nodes
                l_ev = total_rel / n_ev
                e_hit = np.where(n_hit > 0, total_rel / n_hit, np.nan)

            rows.append(
                {
                    "state": state,
                    "year": int(year),
                    "n_sims": int(len(sim)),
                    "n_nodes": n_nodes,
                    "P_hit_mean": float(np.nanmean(p_hit)),
                    "P_hit_std": float(np.nanstd(p_hit, ddof=1)) if len(sim) > 1 else 0.0,
                    "E_loss_given_hit_mean": float(np.nanmean(e_hit)),
                    "E_loss_given_hit_std": float(np.nanstd(e_hit, ddof=1))
                    if np.isfinite(e_hit).sum() > 1
                    else 0.0,
                    "L_event_mean": float(np.nanmean(l_ev)),
                    "L_event_std": float(np.nanstd(l_ev, ddof=1)) if len(sim) > 1 else 0.0,
                    "L_tilde_mean": float(sim["L_tilde"].mean()),
                    "n_events_mean": float(sim["n_events"].mean()),
                    # MC stability diagnostic: CV of L_tilde across sims
                    "L_tilde_cv": float(
                        sim["L_tilde"].std(ddof=1) / abs(sim["L_tilde"].mean())
                    )
                    if abs(sim["L_tilde"].mean()) > EPS and len(sim) > 1
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def pool_states(df: pd.DataFrame) -> pd.DataFrame:
    pooled = (
        df.groupby("state", as_index=False)
        .agg(
            P_hit=("P_hit_mean", "mean"),
            E_loss_given_hit=("E_loss_given_hit_mean", "mean"),
            L_event=("L_event_mean", "mean"),
            L_tilde=("L_tilde_mean", "mean"),
            n_nodes=("n_nodes", "mean"),
            n_events=("n_events_mean", "mean"),
            L_tilde_cv=("L_tilde_cv", "mean"),
            n_sims=("n_sims", "min"),
        )
        .sort_values("L_event", ascending=False)
    )
    # identity check: L_event ≈ P_hit × E[loss|hit]
    pooled["L_event_product"] = pooled["P_hit"] * pooled["E_loss_given_hit"]
    pooled["decomp_rel_err"] = (
        (pooled["L_event"] - pooled["L_event_product"]).abs()
        / pooled["L_event"].clip(lower=EPS)
    )
    return pooled


def attach_quadrants(pooled: pd.DataFrame) -> pd.DataFrame:
    out = pooled.copy()
    if QUADRANT.is_file():
        q = pd.read_csv(QUADRANT)
        cols = [c for c in ("state", "quadrant", "quadrant_label", "rank_L_cum", "rank_L_event") if c in q.columns]
        out = out.merge(q[cols], on="state", how="left")
    else:
        out["quadrant"] = "unknown"
    if POOLED.is_file():
        p = pd.read_csv(POOLED)
        extra = [c for c in ("outage_lambda", "outage_severity", "density", "log_V") if c in p.columns]
        if extra:
            out = out.merge(p[["state"] + extra], on="state", how="left")
    return out


def quadrant_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "quadrant" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("quadrant", as_index=False)
        .agg(
            n=("state", "count"),
            P_hit_median=("P_hit", "median"),
            E_loss_given_hit_median=("E_loss_given_hit", "median"),
            L_event_median=("L_event", "median"),
            P_hit_mean=("P_hit", "mean"),
            E_loss_given_hit_mean=("E_loss_given_hit", "mean"),
        )
        .sort_values("quadrant")
    )


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("P_hit", "L_event", "P(hit) vs L_event"),
        ("E_loss_given_hit", "L_event", "E[loss|hit] vs L_event"),
        ("P_hit", "E_loss_given_hit", "P(hit) vs E[loss|hit]"),
        ("outage_lambda", "P_hit", "λ vs P(hit)"),
        ("outage_severity", "E_loss_given_hit", "severity vs E[loss|hit]"),
        ("density", "E_loss_given_hit", "density vs E[loss|hit]"),
        ("log_V", "P_hit", "log|V| vs P(hit)"),
    ]
    rows = []
    for x, y, label in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        sub = df[[x, y]].dropna()
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub[x], sub[y])
        rows.append(
            {
                "variable_pair": label,
                "x": x,
                "y": y,
                "rho": float(rho),
                "p_value": float(p),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def manuscript_blurb(pooled: pd.DataFrame, qsum: pd.DataFrame) -> str:
    low_hit = pooled.nsmallest(5, "P_hit")[["state", "P_hit", "E_loss_given_hit", "L_event"]]
    high_cond = pooled.nlargest(5, "E_loss_given_hit")[
        ["state", "P_hit", "E_loss_given_hit", "L_event"]
    ]
    unstable = pooled[pooled["L_tilde_cv"].notna()].nlargest(5, "L_tilde_cv")[
        ["state", "L_tilde_cv", "P_hit", "n_sims"]
    ]
    lines = [
        "Section 4.3 main analysis — L_event = P(hit) × E[loss|hit].",
        "P(hit) measures exposure (outage footprint vs station geography);",
        "E[loss|hit] measures conditional structural impact.",
        "Zero-loss events are retained (real outages often miss EV stations).",
        "Panel regressions on severity / topology are supplementary (see 4.3.2 lock).",
        "",
        f"States (n={len(pooled)}). Median P(hit)={pooled['P_hit'].median():.3f}; "
        f"median E[loss|hit]={pooled['E_loss_given_hit'].median():.3e}.",
        "",
        "Lowest exposure (P(hit)):",
        low_hit.to_string(index=False),
        "",
        "Highest conditional structure (E[loss|hit]):",
        high_cond.to_string(index=False),
        "",
        "Highest L_tilde CV across MC sims (consider more sims / seeds if P(hit) low):",
        unstable.to_string(index=False),
    ]
    if not qsum.empty:
        lines += ["", "By chronic/acute quadrant (medians):", qsum.to_string(index=False)]
    lines += [
        "",
        "Interpretation guide: frequency-driven units -> high P(hit);",
        "structurally sensitive units -> high E[loss|hit].",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_decomposition_from_per_sim(RESULTS, YEARS)
    if raw.empty:
        raise SystemExit(f"No per-sim metrics under {RESULTS}")
    raw.to_csv(OUT / "section_4_3_exposure_structure_state_year.csv", index=False)

    pooled = pool_states(raw)
    pooled = attach_quadrants(pooled)
    # ranks
    pooled["rank_P_hit"] = pooled["P_hit"].rank(ascending=False, method="average")
    pooled["rank_E_loss_given_hit"] = pooled["E_loss_given_hit"].rank(
        ascending=False, method="average"
    )
    pooled["rank_L_event_decomp"] = pooled["L_event"].rank(ascending=False, method="average")
    pooled.to_csv(OUT / "section_4_3_exposure_structure_pooled.csv", index=False)

    qsum = quadrant_summary(pooled)
    if not qsum.empty:
        qsum.to_csv(OUT / "section_4_3_exposure_structure_by_quadrant.csv", index=False)

    spears = spearman_table(pooled)
    spears.to_csv(OUT / "section_4_3_exposure_structure_spearman.csv", index=False)

    blurb = manuscript_blurb(pooled, qsum)
    (OUT / "section_4_3_exposure_structure_blurb.txt").write_text(blurb, encoding="utf-8")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "L_event = P(hit) * E[loss|hit]",
        "role": "MAIN section 4.3 analysis; regressions are supplementary",
        "n_states": int(len(pooled)),
        "years": list(YEARS),
        "median_P_hit": float(pooled["P_hit"].median()),
        "median_E_loss_given_hit": float(pooled["E_loss_given_hit"].median()),
        "median_decomp_rel_err": float(pooled["decomp_rel_err"].median()),
        "note_zero_loss": (
            "Zero-loss events are substantive under population-weighted epicenters; "
            "not a bug. If L_tilde CV is high and P(hit) is low, increase n_sims."
        ),
        "supplementary": "section_4_3_2_LOCKED_panel_cluster.md (regression)",
    }
    (OUT / "section_4_3_exposure_structure_CANONICAL.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(blurb)
    print(f"\nWrote outputs under {OUT}")


if __name__ == "__main__":
    main()
