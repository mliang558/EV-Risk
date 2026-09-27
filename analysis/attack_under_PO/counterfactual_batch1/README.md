# Counterfactual Batch 1 (+ A/B) — OAT dose-response package

**Does not modify or overwrite** the original MC pipeline (`monte_carlo_outage_attack.py`, `results_mc_10km_panel_2018_2026/`).

**Reuses** existing 2023 10 km hypernode pickles under `outputs/network_graph_10km_2018_2026/2023/` — CF-D only appends new nodes on top; it does not rebuild the base network.

## Design principles

1. **OAT + CRN:** one intervention factor changes; λ / bootstrap / **epicenter coordinates** drawn once (population-weighted / station-KDE on the **baseline** graph) and reused across scenarios.
2. **Shared dose grid:** 0 / 10 / 20 / 30 / 50% for CF-D rules and CF-S.
3. **CF-D = coverage expansion** (not within-cluster densify). Capacity unused (efficiency is topology-only). Within-cluster add/capacity = beyond model resolution.
4. **Three placement rules** (all enforce ≥ 10 km from existing hypernodes):

| Rule | Family | Method | Role |
|------|--------|--------|------|
| **NEVI corridor** | `CF-D` (main) | Every 50 mi along interstate; drop sites <10 km from existing; take largest gaps first | Policy-relevant |
| **Population-weighted** | `CF-D-pop` | Sample in unit counties ∝ population | Demand-oriented |
| **Uniform null** | `CF-D-null` | Area-weighted in counties; **10 seed replicates** | Spatial control |

5. **CRN epicenter lock:** epicenters are geographic draws (population mode), frozen per sim; densified nodes may fall inside \(R_c\) but **never** change the epicenter. Zero-loss events are expected.
6. **No cost conversion.**
7. **§4.3 decomposition:** report \(L_{\mathrm{event}}=P(\mathrm{hit})\times E[\mathrm{loss}\mid\mathrm{hit}]\) (exposure vs structure); regressions are supplementary.

## Scenario matrix

| Family | Keys | Notes |
|--------|------|-------|
| `baseline` | — | existing 10 km net |
| `CF-D` | `CF-D_nevi_p{10,20,30,50}` | main |
| `CF-D-pop` | `CF-D_pop_p{…}` | optional robustness |
| `CF-D-null` | `CF-D_null_p{…}_r{0..9}` | optional null (40 keys) |
| `CF-S` | `CF-S_m{…}` | −% \(R_c\) |
| `CF-T`, `U`, `K` | … | optional extras |

Default: `FAMILIES=baseline,CF-D,CF-S`. Add `,CF-D-pop,CF-D-null` for full placement sensitivity.

## GIS inputs (reuse local / sibling EV_Project assets)

- Interstate: `../Highway/us_interstate_highways.shp/` (fallback: TIGER PrimaryRoads, `RTTYP=I`)
- County pop: `analysis/attack_under_PO/module_3_03/demographic data/merged_output.csv`
- County polygons: same as MC (`tl_2021_us_county`)

## Cluster

```bash
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"

# Smoke (NEVI main + severity)
ONLY=TX N_SIMS=2 WORKERS=1 FAMILIES=baseline,CF-D,CF-S \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

# Full main OAT
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-S \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

# + placement robustness / null
N_SIMS=100 WORKERS=12 FAMILIES=baseline,CF-D,CF-D-pop,CF-D-null,CF-S \
  bash analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh

python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_cf_batch1_2023_pooled
```

Optional prebuild densified pickles (otherwise built on first use):

```bash
python analysis/attack_under_PO/counterfactual_batch1/build_cf_d_networks.py --only TX
```

## Manuscript wording (suggested)

> We define the network intervention as **coverage expansion**. The primary rule places candidate sites every 50 miles along interstate corridors (NEVI-style), excluding locations within 10 km of an existing hypernode, and retains the largest residual gaps up to each dose. Population-weighted placement and a uniform spatial null (10 replicates) serve as demand-oriented and null controls. Capacity upgrades and within-cluster densification are beyond the 10 km hypernode resolution and are not evaluated. Outage epicenters are sampled once on the baseline network and held fixed under CRN; newly added hypernodes may be disrupted when covered by an impact radius but never serve as epicenters. We compare dose–response slopes of \(\Delta L\) on a common 0–50% intensity grid against severity reduction, without converting interventions to monetary costs.

## Upload

See `UPLOAD_LIST_BATCH1.txt`.
