"""CF-D denominator audit using panel summary only (no metrics pull yet).

W := L_tilde * n_nodes  (= total_rel_loss_x_duration when |V| fixed within scenario)
Does NOT conclude policy — only separates mechanical |V| scaling from residual.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/cf_batch1_2023_pooled")
OUT = ROOT / "analysis_denominator_audit"
OUT.mkdir(parents=True, exist_ok=True)


def f(x):
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def ols_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    # residual SE for slope
    yhat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    df = n - 2
    if df <= 0:
        return slope, intercept, float("nan")
    s2 = ss_res / df
    se = math.sqrt(s2 / sxx)
    return slope, intercept, se


panel = list(csv.DictReader((ROOT / "panel_scenario_summary.csv").open(encoding="utf-8")))

# Enrich with W = L * |V|
rows = []
for r in panel:
    L = f(r["L_tilde_mean"])
    V = f(r["n_nodes_mean"])
    W = L * V if (L == L and V == V) else float("nan")
    rows.append({**r, "W": W, "L": L, "V": V})

by_unit: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    by_unit[r["unit"]].append(r)

# --- 1) Actual |V| growth vs nominal dose x% ---
v_growth = []
for unit, rs in by_unit.items():
    base = next((x for x in rs if x["family"] == "baseline"), None)
    if not base:
        continue
    V0 = base["V"]
    for x in rs:
        if x["family"] != "CF-D":
            continue
        dose = f(x["dose_pct"])
        Vx = x["V"]
        if not (V0 == V0 and Vx == Vx and V0 > 0):
            continue
        growth = Vx / V0 - 1.0
        mech_pct = 100.0 * (V0 / Vx - 1.0)  # L scales as V0/V if W fixed
        v_growth.append(
            {
                "unit": unit,
                "dose_pct": dose,
                "V0": V0,
                "V": Vx,
                "V_growth_frac": growth,
                "mech_L_pct_if_W_fixed": mech_pct,
                "nominal_dose_frac": dose / 100.0,
            }
        )

with (OUT / "cf_d_node_growth.csv").open("w", newline="", encoding="utf-8") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(v_growth[0].keys()))
    w.writeheader()
    w.writerows(v_growth)

print("=== CF-D |V| growth vs nominal dose (cross-unit median) ===")
for dose in (0, 10, 20, 30, 50):
    sub = [r for r in v_growth if abs(r["dose_pct"] - dose) < 1e-9]
    if not sub:
        continue
    g = [r["V_growth_frac"] for r in sub]
    m = [r["mech_L_pct_if_W_fixed"] for r in sub]
    print(
        f"  dose {dose:2d}%: median |V| growth={100*stats.median(g):.1f}% "
        f"(nominal {dose}%) | median pure-denominator L effect={stats.median(m):.1f}% "
        f"| n={len(sub)}"
    )

# --- 2) Pooled dose curves: L vs W (index = 100 * mean/mean0 - 100) ---
def pooled_curve(family: str, metric: str):
    # metric in {L, W}
    out = []
    for dose in (0.0, 10.0, 20.0, 30.0, 50.0):
        pcts = []
        for unit, rs in by_unit.items():
            base = next((x for x in rs if x["family"] == "baseline"), None)
            # dose 0 for CF families: use family dose0 row if present else baseline
            if dose == 0:
                row = next(
                    (x for x in rs if x["family"] == family and abs(f(x["dose_pct"]) - 0) < 1e-9),
                    base,
                )
            else:
                row = next(
                    (
                        x
                        for x in rs
                        if x["family"] == family and abs(f(x["dose_pct"]) - dose) < 1e-9
                    ),
                    None,
                )
            if not base or not row:
                continue
            b = base[metric]
            v = row[metric]
            if not (b == b and v == v) or b == 0:
                continue
            pcts.append(100.0 * (v - b) / abs(b))
        if pcts:
            out.append(
                {
                    "family": family,
                    "dose_pct": dose,
                    "metric": metric,
                    "mean_pct": stats.mean(pcts),
                    "median_pct": stats.median(pcts),
                    "n": len(pcts),
                }
            )
    return out


curves = []
for fam in ("CF-D", "CF-S"):
    curves.extend(pooled_curve(fam, "L"))
    curves.extend(pooled_curve(fam, "W"))

with (OUT / "pooled_dose_L_vs_W.csv").open("w", newline="", encoding="utf-8") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(curves[0].keys()))
    w.writeheader()
    w.writerows(curves)

print("\n=== Pooled mean % change vs baseline ===")
print("dose | CF-D L | CF-D W | CF-S L | CF-S W | CF-D mech(median V)")
for dose in (0, 10, 20, 30, 50):
    def get(fam, met):
        r = next(
            (
                c
                for c in curves
                if c["family"] == fam
                and c["metric"] == met
                and abs(c["dose_pct"] - dose) < 1e-9
            ),
            None,
        )
        return r["mean_pct"] if r else float("nan")

    sub = [r for r in v_growth if abs(r["dose_pct"] - dose) < 1e-9]
    mech = stats.median([r["mech_L_pct_if_W_fixed"] for r in sub]) if sub else float("nan")
    print(
        f"  {dose:2d}% | {get('CF-D','L'):7.2f} | {get('CF-D','W'):7.2f} | "
        f"{get('CF-S','L'):7.2f} | {get('CF-S','W'):7.2f} | {mech:7.2f}"
    )

# Residual after removing mechanical: observed_L_pct - mech_pct (per unit then pool)
print("\n=== CF-D residual after subtracting pure-denominator (per-unit) ===")
resid_pool = defaultdict(list)
for unit, rs in by_unit.items():
    base = next((x for x in rs if x["family"] == "baseline"), None)
    if not base or base["L"] == 0 or not (base["L"] == base["L"]):
        continue
    V0 = base["V"]
    for dose in (10.0, 20.0, 30.0, 50.0):
        row = next(
            (
                x
                for x in rs
                if x["family"] == "CF-D" and abs(f(x["dose_pct"]) - dose) < 1e-9
            ),
            None,
        )
        if not row or not (row["V"] == row["V"]) or row["V"] <= 0:
            continue
        obs = 100.0 * (row["L"] - base["L"]) / abs(base["L"])
        mech = 100.0 * (V0 / row["V"] - 1.0)
        resid_pool[dose].append(obs - mech)
for dose in (10, 20, 30, 50):
    xs = resid_pool[float(dose)]
    print(
        f"  {dose}%: mean residual %ΔL after denom = {stats.mean(xs):.2f} "
        f"(median {stats.median(xs):.2f}), n={len(xs)}"
    )

# --- 3) Per-unit slopes on L% vs W% ---
slope_rows = []
for unit, rs in by_unit.items():
    base = next((x for x in rs if x["family"] == "baseline"), None)
    if not base:
        continue
    for fam in ("CF-D", "CF-S"):
        for metric in ("L", "W"):
            xs, ys = [], []
            for dose in (0.0, 10.0, 20.0, 30.0, 50.0):
                if dose == 0:
                    row = next(
                        (
                            x
                            for x in rs
                            if x["family"] == fam and abs(f(x["dose_pct"]) - 0) < 1e-9
                        ),
                        base,
                    )
                else:
                    row = next(
                        (
                            x
                            for x in rs
                            if x["family"] == fam and abs(f(x["dose_pct"]) - dose) < 1e-9
                        ),
                        None,
                    )
                if not row:
                    continue
                b = base[metric]
                v = row[metric]
                if not (b == b and v == v) or b == 0:
                    continue
                xs.append(dose)
                ys.append(100.0 * (v - b) / abs(b))
            sl, itc, se = ols_slope(xs, ys)
            slope_rows.append(
                {
                    "unit": unit,
                    "family": fam,
                    "metric": metric,
                    "slope": sl,
                    "slope_se": se,
                    "n_points": len(xs),
                }
            )

with (OUT / "slopes_L_vs_W.csv").open("w", newline="", encoding="utf-8") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(slope_rows[0].keys()))
    w.writeheader()
    w.writerows(slope_rows)

# Winner counts on W metric (no CI yet — means only)
wide = defaultdict(dict)
for r in slope_rows:
    wide[r["unit"]][(r["family"], r["metric"])] = r["slope"]

for metric in ("L", "W"):
    s_win = d_win = 0
    pairs = []
    for u, d in wide.items():
        if ("CF-S", metric) in d and ("CF-D", metric) in d:
            s, dd = d[("CF-S", metric)], d[("CF-D", metric)]
            if s != s or dd != dd:
                continue
            pairs.append((u, dd, s))
            if s < dd:
                s_win += 1
            elif dd < s:
                d_win += 1
    print(
        f"\n=== Slope race on {metric} (more neg wins; NO CI yet): "
        f"CF-S {s_win} / CF-D {d_win} / n={len(pairs)}"
    )
    Ss = [s for _, _, s in pairs]
    Ds = [d for _, d, _ in pairs]
    print(f"  mean slope CF-D={stats.mean(Ds):.4f}  CF-S={stats.mean(Ss):.4f}")

# densify meta: n_added / n_base
print("\n=== densify_meta n_added/n_base (from sidecars) ===")
for dose in (10, 20, 30, 50):
    fracs = []
    for p in (ROOT / "units").glob("*/densify_meta.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        key = f"CF-D_nevi_p{dose}"
        if key not in meta:
            continue
        m = meta[key]
        nb, na = m.get("n_base"), m.get("n_added")
        if nb and na is not None and nb > 0:
            fracs.append(na / nb)
    if fracs:
        print(
            f"  p{dose}: median n_added/n_base={100*stats.median(fracs):.1f}% "
            f"mean={100*stats.mean(fracs):.1f}% n={len(fracs)}"
        )

print(f"\nWrote under {OUT}")
print(
    "NOTE: P(hit)/E[loss|hit] split and slope CIs need metrics_*.csv from cluster."
)
print("NOTE: CF-D-pop / CF-D-null were NOT in this run — cannot compare placement rules yet.")
