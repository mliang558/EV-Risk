#!/usr/bin/env python3
"""
Post-run smoke checks for Counterfactual Batch 1 (TX / VT / merged units).

Checks
------
1. CRN: same seed → epicenter (lat,lon) + T_i identical across scenarios;
   dose-0 CF-D / CF-S L_tilde matches baseline.
2. Monotonicity: CF-S L non-increasing in dose; CF-D hit non-decreasing in dose.
3. P(hit) magnitude (warn if ~0 or ~1).
4. K uses 5 km net: |V|_K >> |V|_10km.
5. NEVI meta: n_nevi + n_pop_fill == n_target (== n_added).
6. Optional: merged-unit tract population coverage; fallback events on small states.

Usage
-----
  python .../smoke_test_batch1.py --batch-dir results_cf_batch1_smoke --unit TX
  python .../smoke_test_batch1.py --check-pop-units-only
  python .../smoke_test_batch1.py --batch-dir ... --unit TX --estimate-full \\
      --n-units 45 --n-sims-full 100 --workers 12 --smoke-seconds 600
"""

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

# State FIPS prefixes for merged analysis units in network_summary
MERGED_UNIT_MEMBERS: dict[str, tuple[str, ...]] = {
    "Maryland": ("MD", "DE", "DC"),
    "Connecticut": ("CT", "RI"),
    "Dakotas": ("SD", "ND"),
}
STATE_ABBR_TO_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41",
    "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}

EPS_L = 1e-10
EPS_COORD = 1e-9


class CheckResult:
    def __init__(self) -> None:
        self.ok: list[str] = []
        self.fail: list[str] = []
        self.warn: list[str] = []

    def pass_(self, msg: str) -> None:
        self.ok.append(msg)
        print(f"  [PASS] {msg}")

    def fail_(self, msg: str) -> None:
        self.fail.append(msg)
        print(f"  [FAIL] {msg}")

    def warn_(self, msg: str) -> None:
        self.warn.append(msg)
        print(f"  [WARN] {msg}")

    @property
    def n_fail(self) -> int:
        return len(self.fail)


def _read_events(unit_dir: Path, key: str) -> pd.DataFrame | None:
    p = unit_dir / f"events_{key}.csv.gz"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _read_metrics(unit_dir: Path, key: str) -> pd.DataFrame | None:
    p = unit_dir / f"metrics_{key}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _scenario_keys(unit_dir: Path) -> list[str]:
    keys = []
    for p in sorted(unit_dir.glob("metrics_*.csv")):
        keys.append(p.stem.replace("metrics_", "", 1))
    return keys


def check_crn(unit_dir: Path, r: CheckResult) -> None:
    print("\n[1] CRN lock (epicenter + T_i; dose-0 L)")
    base = _read_events(unit_dir, "baseline")
    if base is None:
        r.fail_("missing events_baseline.csv.gz (need --no-events OFF)")
        return
    keys = [k for k in _scenario_keys(unit_dir) if k != "baseline"]
    if not keys:
        r.fail_("no non-baseline scenarios")
        return

    cols = ["sim_id", "event_id", "epicenter_lat", "epicenter_lon", "duration_hours"]
    for c in cols:
        if c not in base.columns:
            r.fail_(f"baseline events missing column {c}")
            return
    base_idx = base[cols].sort_values(["sim_id", "event_id"]).reset_index(drop=True)

    for key in keys:
        ev = _read_events(unit_dir, key)
        if ev is None:
            r.warn_(f"{key}: no events file — skip CRN event check")
            continue
        for c in cols:
            if c not in ev.columns:
                r.fail_(f"{key}: missing {c}")
                break
        else:
            oth = ev[cols].sort_values(["sim_id", "event_id"]).reset_index(drop=True)
            if len(oth) != len(base_idx):
                r.fail_(
                    f"{key}: n_events={len(oth)} != baseline {len(base_idx)}"
                )
                continue
            lat_ok = np.allclose(
                oth["epicenter_lat"], base_idx["epicenter_lat"], atol=EPS_COORD, equal_nan=True
            )
            lon_ok = np.allclose(
                oth["epicenter_lon"], base_idx["epicenter_lon"], atol=EPS_COORD, equal_nan=True
            )
            # T_i from CRN: prefer duration_base_hours (unscaled); CF-T scales duration_hours
            ev_sorted = ev.sort_values(["sim_id", "event_id"]).reset_index(drop=True)
            if "duration_base_hours" in ev_sorted.columns:
                t_series = ev_sorted["duration_base_hours"]
                t_label = "duration_base_hours"
            else:
                t_series = ev_sorted["duration_hours"]
                t_label = "duration_hours"
            t_ok = np.allclose(
                t_series.to_numpy(),
                base_idx["duration_hours"].to_numpy(),
                atol=EPS_COORD,
                equal_nan=True,
            )
            if lat_ok and lon_ok and t_ok:
                r.pass_(f"{key}: epicenter + {t_label} match baseline (CRN)")
            else:
                r.fail_(
                    f"{key}: CRN mismatch "
                    f"(lat={lat_ok}, lon={lon_ok}, {t_label}={t_ok})"
                )

    # Dose 0 L must equal baseline
    m_base = _read_metrics(unit_dir, "baseline")
    if m_base is None:
        r.fail_("missing metrics_baseline.csv")
        return
    L0 = m_base.set_index("sim_id")["L_tilde"]
    for dose0_key in ("CF-D_nevi_p0", "CF-S_m0"):
        m = _read_metrics(unit_dir, dose0_key)
        if m is None:
            r.warn_(f"{dose0_key} not run — cannot verify dose-0 L lock")
            continue
        Lx = m.set_index("sim_id")["L_tilde"]
        common = L0.index.intersection(Lx.index)
        if len(common) == 0:
            r.fail_(f"{dose0_key}: no overlapping sim_id with baseline")
            continue
        diff = (L0.loc[common] - Lx.loc[common]).abs().max()
        if diff <= EPS_L:
            r.pass_(f"{dose0_key}: L_tilde == baseline (max|Δ|={diff:.2e})")
        else:
            r.fail_(f"{dose0_key}: L_tilde ≠ baseline (max|Δ|={diff:.2e})")


def check_monotonicity(unit_dir: Path, r: CheckResult) -> None:
    print("\n[2] Monotonicity (CF-S L ↓; CF-D hit ↑)")
    # CF-S: mean L_tilde non-increasing in dose
    cfs_keys = sorted(
        [k for k in _scenario_keys(unit_dir) if k.startswith("CF-S_m")],
        key=lambda k: int(k.replace("CF-S_m", "")),
    )
    if len(cfs_keys) >= 2:
        Ls = []
        doses = []
        for k in cfs_keys:
            m = _read_metrics(unit_dir, k)
            if m is None:
                continue
            doses.append(int(k.replace("CF-S_m", "")))
            Ls.append(float(m["L_tilde"].mean()))
        mono = all(Ls[i] + EPS_L >= Ls[i + 1] for i in range(len(Ls) - 1))
        msg = "CF-S L_tilde by dose: " + ", ".join(
            f"{d}%={v:.6g}" for d, v in zip(doses, Ls)
        )
        if mono:
            r.pass_(msg)
        else:
            r.fail_(msg + " — NOT monotone non-increasing")
    else:
        r.warn_("CF-S scenarios missing — skip L monotonicity")

    # CF-D: per-event hit should not decrease as dose increases
    cfd_keys = sorted(
        [k for k in _scenario_keys(unit_dir) if k.startswith("CF-D_nevi_p")],
        key=lambda k: int(k.replace("CF-D_nevi_p", "")),
    )
    if len(cfd_keys) < 2:
        r.warn_("CF-D_nevi scenarios missing — skip hit monotonicity")
        return
    frames = []
    for k in cfd_keys:
        ev = _read_events(unit_dir, k)
        if ev is None or "hit" not in ev.columns:
            r.warn_(f"{k}: no hit column")
            return
        dose = int(k.replace("CF-D_nevi_p", ""))
        frames.append(
            ev[["sim_id", "event_id", "hit"]].rename(columns={"hit": f"hit_{dose}"})
        )
    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on=["sim_id", "event_id"], how="inner")
    hit_cols = [c for c in merged.columns if c.startswith("hit_")]
    doses_h = sorted(int(c.split("_")[1]) for c in hit_cols)
    violations = 0
    n_pairs = 0
    for i in range(len(doses_h) - 1):
        a, b = doses_h[i], doses_h[i + 1]
        ca, cb = f"hit_{a}", f"hit_{b}"
        bad = merged[cb] < merged[ca]
        violations += int(bad.sum())
        n_pairs += len(merged)
    if violations == 0:
        r.pass_(
            f"CF-D hit non-decreasing across doses {doses_h} "
            f"(0 violations / {n_pairs} event-steps)"
        )
    else:
        r.fail_(
            f"CF-D hit decreased on {violations}/{n_pairs} event-steps "
            f"(adding stations must not reduce hit)"
        )


def check_p_hit(unit_dir: Path, r: CheckResult) -> None:
    print("\n[3] P(hit) magnitude")
    m = _read_metrics(unit_dir, "baseline")
    if m is None or "P_hit" not in m.columns:
        # derive from events
        ev = _read_events(unit_dir, "baseline")
        if ev is None or "hit" not in ev.columns:
            r.fail_("cannot compute P(hit)")
            return
        p = float(ev.groupby("sim_id")["hit"].mean().mean())
    else:
        p = float(m["P_hit"].mean())
    n_ev = int(m["n_events"].mean()) if m is not None and "n_events" in m.columns else -1
    msg = f"baseline mean P(hit)={p:.4f}"
    if n_ev > 0:
        msg += f" (mean n_events/sim≈{n_ev:.0f})"
    if p >= 0.95:
        r.fail_(msg + " — near 100%: epicenter likely still hugging stations")
    elif p <= 0.02:
        r.fail_(
            msg + " — near 0%: R_c too small or CRS/epicenter misaligned "
            "(check EPSG:4326 tracts vs station lon/lat)"
        )
    elif p >= 0.85:
        r.warn_(msg + " — high; double-check population epicenters")
    elif p <= 0.05:
        r.warn_(msg + " — low; check radii / county coverage")
    else:
        r.pass_(msg)


def check_k_network(unit_dir: Path, r: CheckResult) -> None:
    print("\n[4] K_5km |V| vs 10 km")
    mb = _read_metrics(unit_dir, "baseline")
    mk = _read_metrics(unit_dir, "K_5km")
    if mk is None:
        r.warn_("K_5km not in this run — skip")
        return
    if mb is None:
        r.fail_("missing baseline metrics for |V| compare")
        return
    n10 = float(mb["n_nodes"].mean())
    n5 = float(mk["n_nodes"].mean())
    ratio = n5 / n10 if n10 > 0 else float("nan")
    msg = f"|V|_10km={n10:.0f}, |V|_5km={n5:.0f}, ratio={ratio:.2f}"
    if n5 > n10 * 1.2:
        r.pass_(msg)
    elif n5 > n10:
        r.warn_(msg + " — only slightly larger; confirm 5 km rebuild")
    else:
        r.fail_(msg + " — K is not larger than 10 km baseline")


def check_nevi_meta(unit_dir: Path, densify_cache: Path | None, r: CheckResult) -> None:
    print("\n[5] NEVI meta: n_nevi + n_pop_fill == n_target")
    meta_path = unit_dir / "densify_meta.json"
    metas: dict = {}
    if meta_path.is_file():
        metas = json.loads(meta_path.read_text(encoding="utf-8"))
    elif densify_cache is not None:
        # fall back to sidecar next to densify pickles
        unit = unit_dir.name
        for p in densify_cache.glob(f"nevi_p*/network_structures/network_{unit}.pkl.meta.json"):
            key_guess = p.parents[1].name  # nevi_p20
            metas[f"CF-D_{key_guess}"] = json.loads(p.read_text(encoding="utf-8"))
        for p in densify_cache.glob(f"nevi_p*/network_structures/network_{unit}.pkl"):
            meta_side = p.with_suffix(".meta.json")
            if meta_side.is_file():
                continue
            try:
                import pickle

                with open(p, "rb") as f:
                    payload = pickle.load(f)
                m = payload.get("meta") or {}
                if m:
                    metas[f"CF-D_{p.parents[1].name}"] = m
            except Exception:
                pass
    if not metas:
        r.warn_("no densify_meta.json — re-run with CF-D or check densify cache")
        return
    any_nevi = False
    for key, m in sorted(metas.items()):
        if m.get("mode") not in (None, "nevi") and "nevi" not in key.lower():
            continue
        if "n_nevi" not in m and "n_target" not in m:
            continue
        any_nevi = True
        n_nevi = int(m.get("n_nevi") or 0)
        n_fill = int(m.get("n_pop_fill") or 0)
        n_tgt = int(m.get("n_target") or m.get("n_added") or -1)
        n_add = int(m.get("n_added") or (n_nevi + n_fill))
        ok = (n_nevi + n_fill == n_add) and (n_tgt < 0 or n_add == n_tgt or n_add <= n_tgt)
        # allow n_add < n_target if both corridor+pop exhausted
        msg = (
            f"{key}: n_nevi={n_nevi} + n_pop_fill={n_fill} = {n_nevi + n_fill} "
            f"(n_added={n_add}, n_target={n_tgt})"
        )
        if n_nevi + n_fill == n_add and (n_tgt < 0 or n_add <= n_tgt):
            if n_add < n_tgt:
                r.warn_(msg + " — shortfall after pop fill (candidates exhausted)")
            else:
                r.pass_(msg)
        else:
            r.fail_(msg)
    if not any_nevi:
        r.warn_("no NEVI entries in densify meta")


def check_fallback(unit_dir: Path, r: CheckResult, expect_fallback: bool) -> None:
    print("\n[6] State-pool fallback (small / merged units)")
    crn = unit_dir / "crn_events.csv.gz"
    if not crn.is_file():
        r.warn_("no crn_events.csv.gz")
        return
    df = pd.read_csv(crn)
    if "used_state_pool_fallback" not in df.columns:
        r.warn_("no used_state_pool_fallback column")
        return
    n_fb = int(df["used_state_pool_fallback"].astype(bool).sum())
    n = len(df)
    msg = f"fallback events={n_fb}/{n} ({100.0 * n_fb / max(n, 1):.1f}%)"
    if expect_fallback and n_fb == 0:
        r.warn_(msg + " — expected some fallback on VT/Dakotas; check outage coverage")
    elif n_fb > 0:
        r.pass_(msg)
    else:
        r.pass_(msg + " — none (OK for large states)")


def check_pop_units_merged(project_root: Path, r: CheckResult) -> None:
    print("\n[pop] Merged-unit census tracts in pop_units_epicenter.gpkg")
    gpkg = project_root / "data" / "processed" / "pop_units_epicenter.gpkg"
    if not gpkg.is_file():
        r.fail_(f"missing {gpkg} — run build_pop_units_epicenter.py first")
        return
    try:
        import geopandas as gpd
    except ImportError:
        r.fail_("geopandas required to check pop units")
        return
    gdf = gpd.read_file(gpkg)
    if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326", "WGS84"):
        r.warn_(f"pop_units CRS={gdf.crs} (expect EPSG:4326)")
    else:
        r.pass_(f"pop_units CRS={gdf.crs}")
    if "county_fips" not in gdf.columns:
        r.fail_("pop_units missing county_fips")
        return
    gdf["county_fips"] = gdf["county_fips"].astype(str).str.zfill(5)
    gdf["st"] = gdf["county_fips"].str[:2]
    for unit, abbrs in MERGED_UNIT_MEMBERS.items():
        missing = []
        for ab in abbrs:
            fips = STATE_ABBR_TO_FIPS[ab]
            sub = gdf[gdf["st"] == fips]
            n = len(sub)
            n_pos = int((sub["population"] > 0).sum()) if "population" in sub.columns else n
            if n == 0:
                missing.append(ab)
                r.fail_(f"{unit}: no tracts for {ab} (FIPS {fips})")
            else:
                r.pass_(f"{unit}: {ab} tracts={n} (pop>0: {n_pos})")
        if not missing:
            r.pass_(f"{unit}: all members present ({','.join(abbrs)})")


def estimate_runtime(
    smoke_seconds: float,
    smoke_n_sims: int,
    n_units: int,
    n_sims_full: int,
    workers: int,
    n_scenarios_smoke: int,
    n_scenarios_full: int | None,
) -> None:
    print("\n[time] Rough wall-clock estimate")
    if smoke_seconds <= 0 or smoke_n_sims <= 0:
        print("  provide --smoke-seconds from the TX smoke wall time")
        return
    sc_full = n_scenarios_full or n_scenarios_smoke
    # scale by sims, scenarios, units / workers
    per_unit_full = smoke_seconds * (n_sims_full / smoke_n_sims) * (
        sc_full / max(n_scenarios_smoke, 1)
    )
    wall_h = (per_unit_full * n_units / max(workers, 1)) / 3600.0
    print(
        f"  smoke: {smoke_seconds:.0f}s for 1 unit × {smoke_n_sims} sims × "
        f"~{n_scenarios_smoke} scenarios"
    )
    print(
        f"  full ≈ {wall_h:.1f} h  "
        f"({n_units} units × {n_sims_full} sims × ~{sc_full} sc / {workers} workers)"
    )
    print(
        "  If too long: drop K and/or U first "
        "(FAMILIES=baseline,CF-D,CF-S), then add back."
    )
    # without K+U: assume ~30% less if those were in smoke
    print(
        f"  without K+U (rough): ~{wall_h * 0.7:.1f} h "
        f"(order-of-magnitude only)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CF Batch1 smoke checks")
    ap.add_argument("--batch-dir", type=str, default="results_cf_batch1_smoke")
    ap.add_argument("--unit", type=str, default="", help="e.g. TX or VT")
    ap.add_argument("--densify-cache", type=str, default="outputs/network_graph_cf_densify_2023")
    ap.add_argument("--project-root", type=str, default=str(ROOT))
    ap.add_argument("--check-pop-units-only", action="store_true")
    ap.add_argument(
        "--expect-fallback",
        action="store_true",
        help="Warn if zero state-pool fallback (use for VT/Dakotas)",
    )
    ap.add_argument("--estimate-full", action="store_true")
    ap.add_argument("--smoke-seconds", type=float, default=0.0)
    ap.add_argument("--smoke-n-sims", type=int, default=2)
    ap.add_argument("--n-units", type=int, default=45)
    ap.add_argument("--n-sims-full", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-scenarios-smoke", type=int, default=12)
    ap.add_argument("--n-scenarios-full", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.project_root)
    r = CheckResult()

    if args.check_pop_units_only or not args.unit:
        check_pop_units_merged(root, r)

    if args.unit:
        batch = Path(args.batch_dir)
        if not batch.is_absolute():
            batch = root / batch
        unit_dir = batch / args.unit
        if not unit_dir.is_dir():
            # try display-name folders
            alts = list(batch.glob(f"*{args.unit}*"))
            if alts:
                unit_dir = alts[0]
            else:
                print(f"ERROR: missing {batch / args.unit}")
                sys.exit(2)
        print(f"Smoke checks: {unit_dir}")
        densify = Path(args.densify_cache)
        if not densify.is_absolute():
            densify = root / densify
        check_crn(unit_dir, r)
        check_monotonicity(unit_dir, r)
        check_p_hit(unit_dir, r)
        check_k_network(unit_dir, r)
        check_nevi_meta(unit_dir, densify, r)
        check_fallback(unit_dir, r, expect_fallback=args.expect_fallback)

    if args.estimate_full:
        estimate_runtime(
            smoke_seconds=args.smoke_seconds,
            smoke_n_sims=args.smoke_n_sims,
            n_units=args.n_units,
            n_sims_full=args.n_sims_full,
            workers=args.workers,
            n_scenarios_smoke=args.n_scenarios_smoke,
            n_scenarios_full=args.n_scenarios_full or None,
        )

    print("\n=== SUMMARY ===")
    print(f"  PASS={len(r.ok)}  WARN={len(r.warn)}  FAIL={len(r.fail)}")
    if r.fail:
        print("FAILED checks:")
        for m in r.fail:
            print(f"  - {m}")
        sys.exit(1)
    print("All hard checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
