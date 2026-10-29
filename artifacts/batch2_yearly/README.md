# Batch 2 yearly robustness artifacts

Source run: `/opt/data_repo/mliang_work/Pro_directory/results_batch2_yearly`.

The files under `analysis_batch2/` are compact analysis outputs only. Raw per-simulation
metrics are not included.

- `panel_unit_year_means.csv`: 45 units x 6 years x 3 designs (810 rows)
- `spearman_by_family_yearpair.csv`: rank persistence with 90% simulation-bootstrap intervals
- `odi_ndi_by_yearpair.csv`: outage/network attribution of rank instability
- `rank_trajectories_L_cum.csv`: unit ranks by design and year
- `fallback_by_unit_year.csv`: severity fallback counts and shares
- `analysis_meta.json`: analysis provenance

Designs:

- A: network year t, outage year t
- B: fixed 2023 network, outage year t
- C: network year t, pooled 2018-2023 outages

These artifacts support the temporal robustness analysis. They do not replace the
canonical pooled Monte Carlo results.
