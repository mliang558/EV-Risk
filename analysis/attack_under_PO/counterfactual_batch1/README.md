# Counterfactual Batch 1 (+ A/B) — OAT dose-response package

**Does not modify or overwrite** the original MC result tree (`results_mc_10km_panel_2018_2026/`).

**Reuses** existing 2023 10 km hypernode pickles under `outputs/network_graph_10km_2018_2026/2023/`.

## Prerequisites (must before any run)

```bash
# Census-tract population units for network-independent epicenters (NO station-KDE)
export CENSUS_API_KEY=...   # recommended
python analysis/attack_under_PO/build_pop_units_epicenter.py
# → data/processed/pop_units_epicenter.gpkg
# Must include tracts for merged units: MD+DE+DC, CT+RI, SD+ND

# Interstate corridors for NEVI CF-D (sibling of Pro_directory):
#   ../Highway/us_interstate_highways.shp/
```

## Design principles

1. **OAT + CRN:** λ / bootstrap / **epicenter coords** drawn once per sim (`seed = sim_id`) and reused.
2. **Epicenter = census-tract population only** (network-independent). Station-KDE is forbidden — it would inflate \(P(\mathrm{hit})\). New CF nodes never become epicenters.
3. **Misses are valid events:** `hit=0`, `ΔE/E=0` — kept in \(P(\mathrm{hit})\) denominator and \(L\) means; excluded only from \(E[\mathrm{loss}\mid\mathrm{hit}]\). Never skip / redraw.
4. **New baseline:** dose 0 uses this epicenter. Dose grid includes **0 / 10 / 20 / 30 / 50%** (`CF-D_nevi_p0`, `CF-S_m0` must match baseline \(L\) under CRN).
5. **CF-S:** \(N_{\mathrm{aff}}-x\%\) ⇒ \(R_c\leftarrow R_c\sqrt{1-x}\).
6. **CF-D = coverage expansion** (NEVI). Shortfall → pop fill; meta: `n_nevi + n_pop_fill = n_added` (≤ `n_target`).
7. **Default families:** `baseline,CF-D,CF-S,U,K`.
8. **§4.3:** \(L_{\mathrm{event}}=P(\mathrm{hit})\times E[\mathrm{loss}\mid\mathrm{hit}]\) is the main analysis.
9. **Appendix epicenter ablation** must use the **same `N_SIMS`** (same seeds) into a separate `OUT`.

## Cluster runbook (smoke → full)

```bash
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"
export CENSUS_API_KEY=...   # if building pop units on cluster

# --- 0) Tract epicenters once ---
python analysis/attack_under_PO/build_pop_units_epicenter.py

# --- 1) SMOKE (TX + VT + auto checks). Must pass before N_SIMS=100 ---
bash analysis/attack_under_PO/counterfactual_batch1/run_smoke_batch1.sh
# Or stepwise:
#   FORCE=1 ONLY=TX N_SIMS=2 WORKERS=1 OUT=results_cf_batch1_smoke \
#     FAMILIES=baseline,CF-D,CF-S,U,K \
#     bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
#   python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
#     --batch-dir results_cf_batch1_smoke --unit TX --estimate-full --smoke-seconds <SEC>
#   FORCE=1 ONLY=VT N_SIMS=2 WORKERS=1 OUT=results_cf_batch1_smoke \
#     FAMILIES=baseline,CF-D,CF-S,U,K \
#     bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
#   python .../smoke_test_batch1.py --batch-dir results_cf_batch1_smoke --unit VT --expect-fallback

# Smoke must confirm:
#   (1) CRN: epicenter + T_i identical; dose-0 L == baseline
#   (2) CF-S L monotone non-increasing; CF-D hit non-decreasing
#   (3) P(hit) not ≈0 / ≈1
#   (4) K |V| ≫ 10 km |V|
#   (5) NEVI meta n_nevi + n_pop_fill = n_added
# VT catches fallback + NEVI shortfall; also checks MD/DE/DC, CT/RI, Dakotas tracts.

# --- 2) FULL (only after smoke PASS) ---
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S,U,K \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

# If wall time too long (see smoke time estimate): drop K+U first
#   N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S bash .../run_batch1_cluster.sh

# --- 3) Appendix: old station epicenter, SAME seeds (same N_SIMS) ---
OUT=results_cf_batch1_2023_pooled_epicenter_station EPICENTER_MODE=station \
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_cf_batch1_2023_pooled
```

## Scenario matrix

| Family | Keys | Notes |
|--------|------|-------|
| `baseline` | — | new epicenter; dose 0 |
| `CF-D` | `CF-D_nevi_p{0,10,20,30,50}` | p0 ≡ baseline under CRN |
| `CF-S` | `CF-S_m{0,10,20,30,50}` | m0 ≡ baseline; \(\sqrt{1-x}\) |
| `U` | `U_x1p5`, `U_x2` | urban \(R_c\) multiplier |
| `K` | `K_5km` | 5 km network |
| `CF-D-pop` / `CF-D-null` | optional | robustness / null |

## Upload

See `UPLOAD_LIST_BATCH1.txt`.
