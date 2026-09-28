#!/usr/bin/env python3
"""
Analyze Batch 2 yearly persistence: Spearman rank ρ across years for A/B/C.

- Adjacent-year ρ and 2018-vs-2023 ρ for L_cum and total_rel_loss_x_duration
- Bootstrap 90% CI over sims
- Per-unit fallback shares
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BATCH = ROOT / "results_batch2_yearly"


def _load_summaries(batch_dir: Path) -> pd.DataFrame:
    rows = []
    for p in batch_dir.glob("*/*/summary.csv"):
        # path: family / netY_outZ / unit / summary.csv
        parts = p.relative_to(batch_dir).parts
        if len(parts) < 3:
            continue
        df = pd.read_csv(p)
        rows.append(df)
    if not rows:
        raise SystemExit(f"No summary.csv under {batch_dir}")
    return pd.concat(rows, ignore_index=True)


def _load_metrics_means(batch_dir: Path) -> pd.DataFrame:
    """One row per family×year_network×year_outage×unit from metrics means."""
    rows = []
    for p in batch_dir.glob("*/*/metrics_baseline.csv"):
        parts = p.relative_to(batch_dir).parts
        family = parts[0]
        tag = parts[1]
        unit = parts[2]
        m = pd.read_csv(p)
        year_network = int(m["year_network"].iloc[0]) if "year_network" in m else np.nan
        yo = m["year_outage"].iloc[0] if "year_outage" in m else tag
        rows.append(
            {
                "family": family,
                "unit": unit,
                "tag": tag,
                "year_network": year_network,
                "year_outage": yo,
                "L_cum": float(m["L_cum"].mean()) if "L_cum" in m else float(m["L_tilde"].mean()),
                "W": float(m["total_rel_loss_x_duration"].mean()),
                "P_hit": float(m["P_hit"].mean()) if "P_hit" in m else np.nan,
                "n_fb_county": int(m["n_fb_county"].sum()) if "n_fb_county" in m else 0,
                "n_fb_state_year": int(m["n_fb_state_year"].sum()) if "n_fb_state_year" in m else 0,
                "n_fb_state_pooled": int(m["n_fb_state_pooled"].sum()) if "n_fb_state_pooled" in m else 0,
                "n_events_tot": int(m["n_events"].sum()) if "n_events" in m else 0,
            }
        )
    return pd.DataFrame(rows)


def spearman_pair(df: pd.DataFrame, y1, y2, metric: str, year_col: str) -> float:
    a = df[df[year_col] == y1][["unit", metric]].rename(columns={metric: "v1"})
    b = df[df[year_col] == y2][["unit", metric]].rename(columns={metric: "v2"})
    m = a.merge(b, on="unit")
    if len(m) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(m["v1"], m["v2"])
    return float(rho)


def bootstrap_spearman(
    batch_dir: Path,
    family: str,
    y1,
    y2,
    metric_col: str,
    year_key: str,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Resample sims with replacement inside each unit×year metrics file,
    recompute unit means, then Spearman across units. 90% CI.
    """
    rng = np.random.default_rng(seed)
    # collect per-unit sim vectors
    store: dict[tuple[str, object], np.ndarray] = {}
    for p in batch_dir.glob(f"{family}/*/metrics_baseline.csv"):
        parts = p.relative_to(batch_dir).parts
        unit = parts[2]
        m = pd.read_csv(p)
        yo = m["year_outage"].iloc[0] if "year_outage" in m else parts[1]
        yn = m["year_network"].iloc[0] if "year_network" in m else np.nan
        key_val = yo if year_key == "year_outage" else yn
        if key_val not in (y1, y2) and str(key_val) not in (str(y1), str(y2)):
            continue
        col = "L_cum" if metric_col == "L_cum" and "L_cum" in m.columns else (
            "L_tilde" if metric_col == "L_cum" else "total_rel_loss_x_duration"
        )
        if col not in m.columns:
            continue
        store[(unit, key_val)] = m[col].to_numpy(dtype=float)

    units = sorted({u for u, _ in store})
    if len(units) < 3:
        return float("nan"), float("nan"), float("nan")

    def rho_once(resamp: bool) -> float:
        v1, v2 = [], []
        for u in units:
            a = store.get((u, y1))
            b = store.get((u, y2))
            if a is None:
                a = store.get((u, str(y1)))
            if b is None:
                b = store.get((u, str(y2)))
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                continue
            if resamp:
                aa = a[rng.integers(0, len(a), size=len(a))]
                bb = b[rng.integers(0, len(b), size=len(b))]
            else:
                aa, bb = a, b
            v1.append(float(np.mean(aa)))
            v2.append(float(np.mean(bb)))
        if len(v1) < 3:
            return float("nan")
        rho, _ = stats.spearmanr(v1, v2)
        return float(rho)

    point = rho_once(False)
    boots = [rho_once(True) for _ in range(n_boot)]
    boots = [x for x in boots if x == x]
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.05, 0.95])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default=str(DEFAULT_BATCH))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    batch = Path(args.batch_dir)
    if not batch.is_absolute():
        batch = ROOT / batch
    out = Path(args.out_dir) if args.out_dir else batch / "analysis_batch2"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    panel = _load_metrics_means(batch)
    panel.to_csv(out / "panel_unit_year_means.csv", index=False)

    # For A/B use year_outage; for C use year_network (pooled outage)
    pairs = [(2018, 2019), (2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2018, 2023)]
    rows = []
    for fam in sorted(panel["family"].dropna().unique()):
        sub = panel[panel["family"] == fam]
        ycol = "year_network" if fam == "C" else "year_outage"
        # coerce numeric years where possible
        sub = sub.copy()
        if ycol == "year_outage":
            sub = sub[sub["year_outage"].astype(str) != "pooled"]
            sub["year_outage"] = pd.to_numeric(sub["year_outage"], errors="coerce")
        for y1, y2 in pairs:
            for metric, mcol in (("L_cum", "L_cum"), ("W", "W")):
                rho = spearman_pair(sub, y1, y2, mcol, ycol)
                point, lo, hi = bootstrap_spearman(
                    batch, fam, y1, y2, mcol, ycol, n_boot=args.n_boot
                )
                rows.append(
                    {
                        "family": fam,
                        "year_a": y1,
                        "year_b": y2,
                        "metric": metric,
                        "spearman_rho": rho,
                        "boot_rho": point,
                        "boot_lo90": lo,
                        "boot_hi90": hi,
                    }
                )
    rho_df = pd.DataFrame(rows)
    rho_df.to_csv(out / "spearman_by_family_yearpair.csv", index=False)

    # fallback rates
    fb = panel.copy()
    tot = (
        fb["n_fb_county"] + fb["n_fb_state_year"] + fb["n_fb_state_pooled"]
    ).replace(0, np.nan)
    fb["share_fb_county"] = fb["n_fb_county"] / tot
    fb["share_fb_state_year"] = fb["n_fb_state_year"] / tot
    fb["share_fb_state_pooled"] = fb["n_fb_state_pooled"] / tot
    fb.to_csv(out / "fallback_by_unit_year.csv", index=False)

    # rank trajectories (L_cum)
    traj_rows = []
    for fam, g in panel.groupby("family"):
        ycol = "year_network" if fam == "C" else "year_outage"
        gg = g.copy()
        if ycol == "year_outage":
            gg = gg[gg["year_outage"].astype(str) != "pooled"]
            gg["year_outage"] = pd.to_numeric(gg["year_outage"], errors="coerce")
        for y, gy in gg.groupby(ycol):
            gy = gy.sort_values("L_cum", ascending=False)
            gy = gy.assign(rank=np.arange(1, len(gy) + 1))
            for _, row in gy.iterrows():
                traj_rows.append(
                    {
                        "family": fam,
                        "year": y,
                        "unit": row["unit"],
                        "L_cum": row["L_cum"],
                        "rank_L_cum": int(row["rank"]),
                    }
                )
    pd.DataFrame(traj_rows).to_csv(out / "rank_trajectories_L_cum.csv", index=False)

    (out / "analysis_meta.json").write_text(
        json.dumps(
            {
                "batch_dir": str(batch),
                "n_boot": args.n_boot,
                "ci": "90% percentile bootstrap over sims",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved under {out}")
    print(rho_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
