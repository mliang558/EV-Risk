#!/usr/bin/env python3
"""
MC noise ceiling / split-half reliability for Batch2 rankings.

No re-run needed: uses existing metrics_baseline.csv (100 sims).

For each (family, year, metric):
  - split sims 1..50 vs 51..100 (by sim_id order)
  - unit means → Spearman ρ between the two half rankings
  - Spearman–Brown: R = 2 r_half / (1 + r_half)  (full-length reliability)

For year-pair Spearman ρ_obs (from analyze_batch2):
  - disattenuate: ρ_corr = ρ_obs / sqrt(R_y1 * R_y2)
  - recompute ODI/NDI from corrected ρ_A, ρ_B, ρ_C

Usage (cluster):
  PYTHONPATH=analysis:analysis/attack_under_PO \\
    python analysis/attack_under_PO/batch2_yearly/check_mc_reliability.py \\
      --batch-dir results_batch2_yearly
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


def spearman_brown(r_half: float) -> float:
    if r_half != r_half or r_half <= -1:
        return float("nan")
    return float(2.0 * r_half / (1.0 + r_half))


def load_sim_panels(batch_dir: Path) -> dict[tuple[str, object, str], pd.DataFrame]:
    """
    Key (family, year_key, metric_col) -> DataFrame[unit, sim_id, value]
    year_key: year_outage for A/B, year_network for C.
    """
    store: dict[tuple[str, object, str], list] = {}
    for p in batch_dir.rglob("metrics_baseline.csv"):
        parts = p.relative_to(batch_dir).parts
        if len(parts) < 4:
            continue
        fam, _tag, unit = parts[0], parts[1], parts[2]
        m = pd.read_csv(p)
        if "sim_id" not in m.columns:
            continue
        yo = m["year_outage"].iloc[0] if "year_outage" in m.columns else None
        yn = m["year_network"].iloc[0] if "year_network" in m.columns else None
        if fam == "C":
            ykey = int(yn) if pd.notna(yn) else None
        else:
            if str(yo) == "pooled":
                continue
            ykey = int(float(yo)) if pd.notna(yo) else None
        if ykey is None:
            continue
        for metric, col in (
            ("L_cum", "L_cum" if "L_cum" in m.columns else "L_tilde"),
            ("W", "total_rel_loss_x_duration"),
        ):
            if col not in m.columns:
                continue
            key = (fam, ykey, metric)
            store.setdefault(key, []).append(
                m[["sim_id", col]].assign(unit=unit).rename(columns={col: "value"})
            )
    out = {}
    for k, parts in store.items():
        out[k] = pd.concat(parts, ignore_index=True)
    return out


def split_half_reliability(df: pd.DataFrame) -> dict:
    """df columns: unit, sim_id, value."""
    d = df.sort_values(["unit", "sim_id"]).copy()
    # stable half split by sim_id rank within unit
    rows = []
    for unit, g in d.groupby("unit"):
        g = g.sort_values("sim_id")
        n = len(g)
        if n < 20:
            continue
        mid = n // 2
        h1 = float(g.iloc[:mid]["value"].mean())
        h2 = float(g.iloc[mid:]["value"].mean())
        rows.append({"unit": unit, "h1": h1, "h2": h2, "n_sims": n})
    if len(rows) < 5:
        return {
            "n_units": len(rows),
            "r_half": float("nan"),
            "R_sb": float("nan"),
        }
    tab = pd.DataFrame(rows)
    r_half = _spearman(tab["h1"].to_numpy(), tab["h2"].to_numpy())
    return {
        "n_units": int(len(tab)),
        "r_half": r_half,
        "R_sb": spearman_brown(r_half),
        "median_n_sims": float(tab["n_sims"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="results_batch2_yearly")
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--spearman-csv",
        default="",
        help="Optional path to spearman_by_family_yearpair.csv (default: batch/analysis_batch2/...)",
    )
    args = ap.parse_args()

    batch = Path(args.batch_dir)
    if not batch.is_absolute():
        batch = ROOT / batch
    out = Path(args.out_dir) if args.out_dir else batch / "analysis_batch2"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading sim-level metrics...", flush=True)
    panels = load_sim_panels(batch)
    print(f"  panels={len(panels)}", flush=True)

    print("[2/3] Split-half reliability by family×year×metric...", flush=True)
    rel_rows = []
    for (fam, ykey, metric), df in sorted(panels.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        r = split_half_reliability(df)
        rel_rows.append(
            {
                "family": fam,
                "year": ykey,
                "metric": metric,
                **r,
            }
        )
    rel_df = pd.DataFrame(rel_rows)
    rel_path = out / "mc_split_half_reliability.csv"
    rel_df.to_csv(rel_path, index=False)

    print("\n=== Reliability summary (Spearman–Brown R) ===")
    for fam in ("A", "B", "C"):
        sub = rel_df[rel_df["family"] == fam]
        for metric in ("W", "L_cum"):
            s = sub[sub["metric"] == metric]["R_sb"]
            if s.empty:
                continue
            print(
                f"  {fam} {metric}: median R={s.median():.3f} "
                f"min={s.min():.3f} max={s.max():.3f} n_years={s.notna().sum()}"
            )

    spearman_path = (
        Path(args.spearman_csv)
        if args.spearman_csv
        else out / "spearman_by_family_yearpair.csv"
    )
    if not spearman_path.is_file():
        print(f"\n[warn] missing {spearman_path}; skip disattenuation / ODI")
        (out / "mc_reliability_meta.json").write_text(
            json.dumps({"rel_csv": str(rel_path), "rule": "R>=0.9 ok; else disattenuate"}, indent=2),
            encoding="utf-8",
        )
        return

    print("[3/3] Disattenuate year-pair ρ and recompute ODI...", flush=True)
    rho_df = pd.read_csv(spearman_path)
    # reliability lookup
    R = {
        (r["family"], int(r["year"]), r["metric"]): float(r["R_sb"])
        for _, r in rel_df.iterrows()
        if r["R_sb"] == r["R_sb"]
    }

    corr_rows = []
    for _, row in rho_df.iterrows():
        fam = row["family"]
        y1, y2 = int(row["year_a"]), int(row["year_b"])
        metric = row["metric"]
        rho_obs = float(row["spearman_rho"])
        R1 = R.get((fam, y1, metric), float("nan"))
        R2 = R.get((fam, y2, metric), float("nan"))
        if rho_obs != rho_obs or R1 != R1 or R2 != R2 or R1 <= 0 or R2 <= 0:
            rho_corr = float("nan")
            rho_corr_clip = float("nan")
        else:
            rho_corr = float(rho_obs / np.sqrt(R1 * R2))
            rho_corr_clip = float(np.clip(rho_corr, -1.0, 1.0))
        corr_rows.append(
            {
                "family": fam,
                "year_a": y1,
                "year_b": y2,
                "metric": metric,
                "rho_obs": rho_obs,
                "R_year_a": R1,
                "R_year_b": R2,
                "rho_disattenuated": rho_corr,
                "rho_disattenuated_clip": rho_corr_clip,
                "min_R": float(np.nanmin([R1, R2])),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "spearman_disattenuated.csv", index=False)

    odi_rows = []
    for (y1, y2, metric), g in corr_df.groupby(["year_a", "year_b", "metric"]):
        piv_obs = {r["family"]: r["rho_obs"] for _, r in g.iterrows()}
        piv_c = {r["family"]: r["rho_disattenuated_clip"] for _, r in g.iterrows()}
        if not all(f in piv_obs for f in ("A", "B", "C")):
            continue
        if not all(piv_c.get(f) == piv_c.get(f) for f in ("A", "B", "C")):
            continue
        u_b = 1.0 - float(piv_c["B"])
        u_c = 1.0 - float(piv_c["C"])
        denom = u_b + u_c
        odi_rows.append(
            {
                "year_a": y1,
                "year_b": y2,
                "metric": metric,
                "rho_A_obs": piv_obs["A"],
                "rho_B_obs": piv_obs["B"],
                "rho_C_obs": piv_obs["C"],
                "rho_A_corr": piv_c["A"],
                "rho_B_corr": piv_c["B"],
                "rho_C_corr": piv_c["C"],
                "ODI_obs": (1 - piv_obs["B"]) / ((1 - piv_obs["B"]) + (1 - piv_obs["C"]))
                if ((1 - piv_obs["B"]) + (1 - piv_obs["C"])) > 1e-12
                else float("nan"),
                "NDI_obs": (1 - piv_obs["C"]) / ((1 - piv_obs["B"]) + (1 - piv_obs["C"]))
                if ((1 - piv_obs["B"]) + (1 - piv_obs["C"])) > 1e-12
                else float("nan"),
                "ODI_corr": u_b / denom if denom > 1e-12 else float("nan"),
                "NDI_corr": u_c / denom if denom > 1e-12 else float("nan"),
                "min_R_ABC": float(
                    np.nanmin(
                        [
                            g.loc[g["family"] == f, "min_R"].iloc[0]
                            for f in ("A", "B", "C")
                        ]
                    )
                ),
            }
        )
    odi_df = pd.DataFrame(odi_rows)
    odi_df.to_csv(out / "odi_ndi_reliability_adjusted.csv", index=False)

    # Decision helper
    print("\n=== Decision (W, 2018–2023) ===")
    focus = odi_df[(odi_df["year_a"] == 2018) & (odi_df["year_b"] == 2023) & (odi_df["metric"] == "W")]
    rel_w = rel_df[rel_df["metric"] == "W"]
    print(
        f"  median R by family: "
        + ", ".join(
            f"{f}={rel_w[rel_w.family==f]['R_sb'].median():.3f}" for f in ("A", "B", "C")
        )
    )
    if not focus.empty:
        r = focus.iloc[0]
        print(
            f"  rho_A obs→corr: {r['rho_A_obs']:.3f}→{r['rho_A_corr']:.3f}; "
            f"ODI obs→corr: {r['ODI_obs']:.3f}→{r['ODI_corr']:.3f}; "
            f"min_R={r['min_R_ABC']:.3f}"
        )
        if r["min_R_ABC"] >= 0.9:
            print("  VERDICT: reliability high (≥0.9) → report raw ρ / ODI.")
        else:
            print("  VERDICT: reliability below 0.9 → report disattenuated ρ / ODI_corr.")

    (out / "mc_reliability_meta.json").write_text(
        json.dumps(
            {
                "split": "first half vs second half of sims by sim_id within unit",
                "spearman_brown": "R=2*r_half/(1+r_half)",
                "disattenuation": "rho_corr=rho_obs/sqrt(R_y1*R_y2)",
                "threshold": 0.9,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved under {out}")
    print(f"  {rel_path.name}")
    print("  spearman_disattenuated.csv")
    print("  odi_ndi_reliability_adjusted.csv")


if __name__ == "__main__":
    main()
