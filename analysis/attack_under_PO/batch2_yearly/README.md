# Batch 2 — year-specific outage persistence (A / B / C)

Baseline only. Reuses Batch-1 loss evaluation, epicenter (tract population),
removal cache `(graph_id, frozenset(S))`, and metrics. New package:
`analysis/attack_under_PO/batch2_yearly/`. Outputs: `results_batch2_yearly/`.

## Designs

| Family | Network | Outages | λ |
|--------|---------|---------|---|
| **A** | year t | year t | `λ_c,t = λ_c × r[s,t]` |
| **B** | **2023** | year t | same r scaling |
| **C** | year t | **2018–2023 pooled** | unscaled `λ_c` (Batch-1-like) |

- `r[s,t] = N[s,t] / mean_t N[s,t]` with `mean_t r = 1`
- **A_2023 ≡ B_2023**: run once under A, copy to B, assert metrics match
- B years when A is included: **2018–2022** only (2023 shared)

## Seeds (CRN)

- **A/B:** `md5(unit|year_outage|sim)[:8]` — network year independent ⇒ A/B same events
- **C:** `sim_id` — matches Batch-1 baseline

## Severity fallback

1. `county_year_pool[c,t]`
2. `state_year_pool[s,t]`
3. 6-year state pool (flag `state_pooled`)

## Smoke → full

```bash
cd /opt/data_repo/mliang_work/Pro_directory
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# smoke (in-memory asserts + optional short run)
FORCE=1 ONLY=VT,TX N_SIMS=2 FAMILIES="A B C" \
  OUT=$PWD/results_batch2_yearly_smoke \
  bash analysis/attack_under_PO/batch2_yearly/run_cluster.sh
python analysis/attack_under_PO/batch2_yearly/smoke_test.py \
  --batch-dir results_batch2_yearly_smoke --units VT,TX

# full
N_SIMS=100 WORKERS=45 NO_EVENTS=1 FAMILIES="A B C" \
  OUT=$PWD/results_batch2_yearly \
  bash analysis/attack_under_PO/batch2_yearly/run_cluster.sh

python analysis/attack_under_PO/batch2_yearly/analyze_batch2.py \
  --batch-dir results_batch2_yearly
```

## Layout

```
results_batch2_yearly/
  r_st_table.csv
  A/net2018_out2018/<unit>/metrics_baseline.csv
  B/net2023_out2018/<unit>/...
  B/net2023_out2023/<unit>/...   # copy of A_2023
  C/net2018_pooled/<unit>/...
  analysis_batch2/
```
