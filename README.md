# EV charging vulnerability under power outages

This repository contains the network construction, outage Monte Carlo simulation, and
robustness analyses used in the manuscript.

## Active manuscript workflow

The current paper uses one canonical analysis and two focused checks.

| Analysis | Question | Status |
| --- | --- | --- |
| Canonical pooled analysis | Which analytical units have the highest long-run outage vulnerability? | Main result; unchanged |
| Batch 2 yearly robustness | Does the ranking change with network year or outage year? | Required robustness check |
| Batch 1 intervention | Does NEVI-style node expansion reduce loss? | Optional policy/appendix analysis |

The detailed code and result audit is in
[`docs/ANALYSIS_REVIEW_2026-10-28.md`](docs/ANALYSIS_REVIEW_2026-10-28.md).

## One cluster entry point

From the project root on the cluster, with the project environment activated:

```bash
pip install -r analysis/attack_under_PO/counterfactual_batch1/requirements-batch1.txt

# Verify the environment and run both smoke tests.
bash analysis/attack_under_PO/run_review_experiments_cluster.sh smoke

# Run only the temporal robustness experiment.
N_SIMS=100 WORKERS=45 \
  bash analysis/attack_under_PO/run_review_experiments_cluster.sh temporal

# Run only the NEVI-style intervention and severity comparator.
N_SIMS=100 WORKERS=45 \
  bash analysis/attack_under_PO/run_review_experiments_cluster.sh intervention

# Analyze completed outputs, including the node-denominator audit and MC reliability.
bash analysis/attack_under_PO/run_review_experiments_cluster.sh analyze
```

The optional placement controls are intentionally separate because the uniform null uses
10 placement replicates at every positive dose:

```bash
N_SIMS=100 WORKERS=45 \
  bash analysis/attack_under_PO/run_review_experiments_cluster.sh placement
```

The central script writes under `results_review/` and does not overwrite the canonical
result tree.

## Method conventions that must match the manuscript

- Charging stations are grouped into hypernodes using the stated spatial clustering
  radius.
- Voronoi/Delaunay construction determines spatial adjacency.
- The outage metric uses NetworkX unweighted global efficiency. Geographic distance
  affects adjacency, but stored edge-distance weights do not enter shortest paths.
- The main loss is node-normalized. Intervention results also report total loss without
  the scenario node-count denominator.
- Batch 1 uses common random numbers across scenarios and population-based epicenters.
- NEVI-style expansion places candidates along interstate corridors; it is not a claim
  of exact regulatory NEVI compliance.

## Active code

```text
analysis/attack_under_PO/
├── monte_carlo_outage_attack.py          # canonical outage simulation core
├── mc_panel_metrics.py                   # canonical panel metrics
├── counterfactual_batch1/                # intervention and placement controls
├── batch2_yearly/                        # temporal robustness
└── run_review_experiments_cluster.sh     # central cluster entry point
```

The directories named `old/` and `module_3_03/`, plus exploratory GNN scripts, are not
part of the active manuscript workflow. They remain in the tree only for provenance or
legacy data-path compatibility and should not be used as analysis entry points.

## Review artifacts

```text
artifacts/
├── cf_batch1_2023_pooled/       # compact intervention results and denominator audit
└── batch2_yearly/               # compact temporal-robustness analysis outputs
```

Raw data, network pickles, and full per-simulation result trees remain on the cluster and
are not duplicated in Git.

## Environment

The review pipeline requires Python 3 with NumPy, pandas, SciPy, NetworkX, GeoPandas,
Shapely, PyProj, xarray, netCDF4, ArviZ, Matplotlib, and PyArrow. Use the pinned minimum
requirements in `counterfactual_batch1/requirements-batch1.txt` and run the environment
check before launching cluster jobs.

## License

Research use; see [`LICENSE`](LICENSE).
