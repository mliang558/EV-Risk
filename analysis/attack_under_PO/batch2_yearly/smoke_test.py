#!/usr/bin/env python3
"""Smoke tests for Batch 2 (VT + TX, N_SIMS=2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ATTACK = Path(__file__).resolve().parents[1]
for p in (ROOT, ATTACK):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from batch2_yearly.rates import build_r_table, ensure_outage_year  # noqa: E402
from batch2_yearly.sampling import crn_seed  # noqa: E402
from batch2_yearly.engine import draw_crn_events_batch2  # noqa: E402
from batch2_yearly.run_parallel import _out_path  # noqa: E402
from monte_carlo_outage_attack import (  # noqa: E402
    load_county_geometry_and_mcc,
    load_county_polygons,
    load_coverage_history,
)
from sample_lambda_from_posterior import load_posterior  # noqa: E402
from select_epicenter_by_population import require_pop_units  # noqa: E402


class R:
    def __init__(self) -> None:
        self.failed = False

    def fail(self, msg: str) -> None:
        print(f"FAIL: {msg}")
        self.failed = True

    def ok(self, msg: str) -> None:
        print(f"PASS: {msg}")

    def warn(self, msg: str) -> None:
        print(f"WARN: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="results_batch2_yearly_smoke")
    ap.add_argument("--units", default="VT,TX")
    ap.add_argument("--n-sims", type=int, default=2)
    args = ap.parse_args()
    r = R()

    batch = Path(args.batch_dir)
    if not batch.is_absolute():
        batch = ROOT / batch

    outages = ensure_outage_year(
        pd.read_csv(
            ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
        )
    )
    r_table = build_r_table(outages)
    # mean_t r == 1
    means = r_table.groupby("state")["r"].mean()
    if not ((means - 1.0).abs() < 1e-8).all():
        r.fail("mean_t r[s,t] != 1")
    else:
        r.ok("mean_t r[s,t] == 1 for all states")

    # print N and r for smoke units' states
    from region_merge_config import ABBR_TO_STATE_NAME

    for ab in args.units.split(","):
        ab = ab.strip()
        st = ABBR_TO_STATE_NAME.get(ab, ab)
        sub = r_table[r_table["state"] == st].sort_values("year")
        print(f"\n[{ab}/{st}] N[s,t] and r[s,t]:")
        print(sub[["year", "N", "r"]].to_string(index=False))

    # A/B event identity for one sim (in-memory, no full attack)
    county_post_df, flat_lam = load_posterior()
    county_geom_mcc = load_county_geometry_and_mcc(ROOT)
    cov_df = load_coverage_history(ROOT)
    polys = load_county_polygons(ROOT)
    poly_dict = {row["fips_str"]: row["geometry"] for _, row in polys.iterrows()}
    pop_gdf = require_pop_units(ROOT)

    for ab in args.units.split(","):
        ab = ab.strip()
        st = ABBR_TO_STATE_NAME.get(ab, ab)
        t = 2020
        for sim in range(args.n_sims):
            kwargs = dict(
                unit=ab,
                sim_id=sim,
                year_outage=t,
                target_states=[st],
                outages_all=outages,
                r_table=r_table,
                county_post_df=county_post_df,
                flat_lam=flat_lam,
                county_geom_mcc=county_geom_mcc,
                cov_df=cov_df,
                poly_dict=poly_dict,
                pop_gdf=pop_gdf,
            )
            a = draw_crn_events_batch2(family="A", **kwargs)
            b = draw_crn_events_batch2(family="B", **kwargs)
            if a.empty or b.empty:
                r.warn(f"{ab} sim{sim}: empty events")
                continue
            cols = ["epicenter_lat", "epicenter_lon", "duration_hours", "radius_base_km"]
            if not np.allclose(a[cols].to_numpy(), b[cols].to_numpy(), equal_nan=True):
                r.fail(f"{ab} sim{sim}: A/B event sequence mismatch")
            else:
                r.ok(f"{ab} sim{sim}: A/B CRN events identical (seed={crn_seed(family='A', unit=ab, year_outage=t, sim_id=sim)})")

    # If smoke run dirs exist, check A_2023 == B_2023 files
    for ab in args.units.split(","):
        ab = ab.strip()
        a_dir = _out_path(batch, "A", 2023, 2023, ab)
        b_dir = _out_path(batch, "B", 2023, 2023, ab)
        if a_dir.is_dir() and b_dir.is_dir():
            am = pd.read_csv(a_dir / "metrics_baseline.csv")
            bm = pd.read_csv(b_dir / "metrics_baseline.csv")
            for c in ("L_cum", "total_rel_loss_x_duration", "P_hit"):
                if c in am.columns and c in bm.columns:
                    if not np.allclose(am[c], bm[c], equal_nan=True):
                        r.fail(f"{ab} A_2023 vs B_2023 {c} mismatch")
                    else:
                        r.ok(f"{ab} A_2023 == B_2023 on {c}")
        else:
            r.warn(f"{ab}: no on-disk A/B 2023 smoke dirs yet (run cluster smoke first)")

    # fallback share sanity on in-memory A draw
    for ab in args.units.split(","):
        ab = ab.strip()
        st = ABBR_TO_STATE_NAME.get(ab, ab)
        ev = draw_crn_events_batch2(
            family="A",
            unit=ab,
            sim_id=0,
            year_outage=2019,
            target_states=[st],
            outages_all=outages,
            r_table=r_table,
            county_post_df=county_post_df,
            flat_lam=flat_lam,
            county_geom_mcc=county_geom_mcc,
            cov_df=cov_df,
            poly_dict=poly_dict,
            pop_gdf=pop_gdf,
        )
        if ev.empty or "fallback_level" not in ev.columns:
            continue
        vc = ev["fallback_level"].value_counts(normalize=True)
        print(f"[{ab}] fallback mix sim0/2019:\n{vc.to_string()}")
        if float(vc.get("state_pooled", 0)) > 0.5:
            r.warn(f"{ab}: >50% state_pooled fallback")

    if r.failed:
        raise SystemExit(1)
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
