# Counterfactual Batch 1 (+ A/B) — OAT dose-response package

**Does not modify or overwrite** the original MC result tree (`results_mc_10km_panel_2018_2026/`).

**Reuses** existing 2023 10 km hypernode pickles under `outputs/network_graph_10km_2018_2026/2023/`.

## Prerequisites (must before any run)

```bash
# Census-tract population units for network-independent epicenters (NO station-KDE)
export CENSUS_API_KEY=...   # recommended
python analysis/attack_under_PO/build_pop_units_epicenter.py
# → data/processed/pop_units_epicenter.gpkg

# Interstate corridors for NEVI CF-D (sibling of Pro_directory):
#   ../Highway/us_interstate_highways.shp/
```

## Design principles

1. **OAT + CRN:** λ / bootstrap / **epicenter coords** drawn once per sim and reused.
2. **Epicenter = census-tract population only** (network-independent). Station-KDE is forbidden — it would inflate \(P(\mathrm{hit})\). New CF nodes never become epicenters.
3. **New baseline:** dose 0 uses this epicenter mechanism. All \(\Delta L\) are vs that baseline. Appendix: re-run with `--epicenter-mode station` into a separate out dir for old-vs-new comparison.
4. **Shared dose grid:** 0 / 10 / 20 / 30 / 50% for CF-D and CF-S.
5. **CF-S locked definition:** cut **affected customers** by \(x\%\) ⇒ \(R_c \leftarrow R_c\sqrt{1-x}\) (area ∝ \(N\)). Not a direct \(R_c\times(1-x)\) cut.
6. **CF-D = coverage expansion** (NEVI main). Dose = \(\lceil pct\cdot|V|\rceil\). If corridor candidates run out, fill with population-weighted sites; meta records `n_nevi`, `n_pop_fill`.
7. **Event log:** each event stores `hit`, `n_disrupted_nodes` (\(|S_i|`), `pct_eff_loss` (\(\Delta E/E\)), `duration_hours` (\(T_i\)), epicenter lat/lon, `seed`.
8. **Default families:** `baseline,CF-D,CF-S,U,K` (U/K share the same CRN cheaply).
9. **§4.3:** \(L_{\mathrm{event}}=P(\mathrm{hit})\times E[\mathrm{loss}\mid\mathrm{hit}]\) is the main analysis.
10. **Batch A/B** and **§4.5 validation** must use the same tract-population epicenter mechanism.

## Scenario matrix

| Family | Keys | Notes |
|--------|------|-------|
| `baseline` | — | new epicenter; dose 0 |
| `CF-D` | `CF-D_nevi_p{10,20,30,50}` | NEVI + pop fill if short |
| `CF-S` | `CF-S_m{…}` | \(N_{\mathrm{aff}}-x\%\) via \(\sqrt{1-x}\) on \(R_c\) |
| `U` | `U_x1p5`, `U_x2` | urban \(R_c\) multiplier |
| `K` | `K_5km` | 5 km network (build on demand) |
| `CF-D-pop` / `CF-D-null` | optional | robustness / null |

## Cluster

```bash
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"
export CENSUS_API_KEY=...   # if building pop units on cluster

# Build tract epicenter units once
python analysis/attack_under_PO/build_pop_units_epicenter.py

# Smoke
ONLY=TX N_SIMS=2 WORKERS=1 FAMILIES=baseline,CF-D,CF-S,U,K \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

# Full (main)
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S,U,K \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

# Appendix: old station epicenter vs new (separate out dir)
OUT=results_cf_batch1_2023_pooled_epicenter_station EPICENTER_MODE=station \
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
# (wire EPICENTER_MODE through runner if needed — or pass via python --epicenter-mode)

python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_cf_batch1_2023_pooled
```

## Upload

See `UPLOAD_LIST_BATCH1.txt`.
