# Counterfactual Batch 1 (+ A/B) — OAT dose-response package

**Does not modify or overwrite** the original MC result tree (`results_mc_10km_panel_2018_2026/`).

**Reuses** existing 2023 10 km hypernode pickles under `outputs/network_graph_10km_2018_2026/2023/`.

## Environment (install first on cluster)

```bash
# From Pro_directory root, with your venv activated:
pip install -r analysis/attack_under_PO/counterfactual_batch1/requirements-batch1.txt

# If geopandas fails on pip wheels:
#   conda install -c conda-forge geopandas shapely pyproj fiona rtree

# Quick import check
python -c "import geopandas, networkx, xarray, netCDF4, shapely; print('ok')"
```

Main packages: `numpy`, `pandas`, `scipy`, `networkx`, `geopandas` (+ `shapely`/`pyproj`/`fiona`), `xarray`, `netCDF4`, `matplotlib`, `pyarrow`.

Also set:

```bash
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO:$PYTHONPATH"
# Shell scripts from Windows may need: sed -i 's/\r$//' analysis/attack_under_PO/counterfactual_batch1/*.sh
```

## Prerequisites (data; must before any run)

**Geography:** see `../GEOGRAPHY_FIPS_LOCK.md` — county shp = **tl_2021** (legacy CT
FIPS matching EAGLE-I/MCC); tracts = **TIGER 2020** + 2020 census (or ACS on 2020 GEOID).
Do not swap in 2023 county boundaries (CT planning regions break CT+RI).

```bash
# Census-tract population units for network-independent epicenters (NO station-KDE)
export CENSUS_API_KEY=...   # required unless strata CSV has filled population
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

# --- 0) Env + tract epicenters once ---
pip install -r analysis/attack_under_PO/counterfactual_batch1/requirements-batch1.txt
python analysis/attack_under_PO/build_pop_units_epicenter.py

# --- 1) SMOKE (TX + VT + auto checks). Must pass before N_SIMS=100 ---
bash analysis/attack_under_PO/counterfactual_batch1/run_smoke_batch1.sh
# Or stepwise (absolute OUT if layout is nonstandard):
#   FORCE=1 ONLY=TX N_SIMS=2 WORKERS=1 \
#     OUT=$PWD/results_cf_batch1_smoke FAMILIES=baseline,CF-D,CF-S,U,K \
#     bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh
#   python analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
#     --batch-dir results_cf_batch1_smoke --unit TX --estimate-full --smoke-seconds <SEC>
#   FORCE=1 ONLY=VT N_SIMS=2 WORKERS=1 OUT=$PWD/results_cf_batch1_smoke \
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

# --- 1b) Removal-cache correctness (VT, ~minutes) ---
# Keys are (graph_id, frozenset(S)); LRU via CF_B1_REMOVAL_CACHE_MAX (default 50000).
# Disable cache: CF_B1_REMOVAL_CACHE=0
python analysis/attack_under_PO/counterfactual_batch1/smoke_compare_removal_cache.py \
  --unit VT --n-sims 2
# Expect: PASS … metric rows identical (cache ON == OFF)
# Optional CA memory peek: same script --unit CA (watch RSS / removal_cache_entries in manifest)

# Clear interrupted partials before full run (must match OUT below):
#   pkill -f run_batch1 || true
#   rm -rf "$PWD/results_cf_batch1_2023_pooled" "$PWD/results_cf_batch1_cache_compare"

# --- 2) FULL (only after smoke + cache-compare PASS) ---
# Use tmux/nohup — long wall time; SSH drop must not kill the job.
# Threads: one BLAS thread per worker process (45 workers × multi-thread = thrash).
tmux new -s cf1   # or: tmux attach -t cf1
# inside tmux:
cd /opt/data_repo/mliang_work/Pro_directory   # your cluster root
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=$PWD/.mplconfig XDG_CACHE_HOME=$PWD/.cache

N_SIMS=100 WORKERS=45 NO_EVENTS=1 \
  OUT=$PWD/results_cf_batch1_2023_pooled \
  FAMILIES=baseline,CF-D,CF-S,U,K \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh \
  2>&1 | tee cf1.log

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
