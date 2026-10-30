# Batch 1 intervention experiment

This package tests a mitigation question using the 2023 network, pooled 2018-2023
outages, and common random numbers across scenarios. It does not replace or modify the
canonical state-ranking analysis.

## Retained design

| Family | Doses | Role |
| --- | --- | --- |
| `baseline` | 0 | Reference |
| `CF-D` | 0, 10, 20, 30, 50% | NEVI-style Interstate corridor node expansion |
| `CF-S` | 0, 10, 20, 30, 50% | Reduction in customers affected per event |
| `CF-D-pop` | 10, 20, 30, 50% | Optional population-placement appendix control |
| `CF-D-null` | 10, 20, 30, 50%, 10 replicates | Optional uniform-placement appendix control |

The default run is `baseline,CF-D,CF-S`. Population and uniform controls must be
requested explicitly. Urban-radius, 5 km network, within-cluster, and duration scenarios
are not part of the active manuscript workflow.

## Environment

```bash
cd /opt/data_repo/mliang_work/Pro_directory
pip install -r analysis/attack_under_PO/counterfactual_batch1/requirements-batch1.txt
python -c "import arviz, geopandas, networkx, xarray, netCDF4, shapely, scipy, pandas, numpy; print('environment OK')"
export PYTHONPATH="$PWD/analysis:$PWD/analysis/attack_under_PO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The population epicenter file and interstate shapefile must already exist. See
`../GEOGRAPHY_FIPS_LOCK.md` for the locked geography definitions.

## Recommended execution

Run the two retained intervention experiments directly with Python. The same
pooled Bayesian outage draw is reused across the baseline and every dose in a
Monte Carlo replication.

```bash
python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py --families baseline,CF-D,CF-S --check-only

python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py --network-root outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023 --network-year 2023 --densify-cache outputs/1029-network-nevi-expansion-2023 --out results_interventions_1029_network2023_pooled2018_2023 --families baseline,CF-D,CF-S --n-sims 100 --workers 45 --no-events
```

For a quick TX smoke test, add `--only TX --n-sims 2 --workers 1` and use a
separate output directory.

## Analysis

```bash
python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_interventions_1029_network2023_pooled2018_2023

python analysis/attack_under_PO/counterfactual_batch1/audit_cf_d_denominator.py \
  --batch-dir results_interventions_1029_network2023_pooled2018_2023
```

The first command requires exactly five unique doses for CF-D and CF-S. The second
reports both node-normalized loss `L` and total loss `W = L * scenario node count`.

## Interpretation locks

- The graph is spatially constructed but global efficiency is unweighted.
- CF-D means NEVI-style corridor expansion, not exact NEVI compliance.
- New nodes are outage targets but never outage epicenters.
- CF-D never fills corridor shortfalls with off-corridor population sites;
  achieved node additions and shortfalls are recorded for every unit and dose.
- Report both `L` and `W`; adding nodes mechanically changes the denominator of `L`.
- Placement controls support a relative comparison only after their full results exist.
