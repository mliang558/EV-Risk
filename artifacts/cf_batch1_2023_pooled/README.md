# CF Batch1 artifacts (2023 × pooled) — NOT the main MC panel

- Source run: `results_cf_batch1_2023_pooled/` on cluster
- Design: OAT + CRN, epicenter=population, N_SIMS=100, families baseline/CF-D/CF-S/U/K
- Do **not** confuse with `results_mc_10km_panel_2018_2026/` (manuscript MC)

The active manuscript review uses only baseline, CF-D, and CF-S rows from this historical
run. U and K are retained in the raw panel for provenance but are not part of the current
argument.

- `analysis_dose_response/`: corrected five-dose CF-D/CF-S slopes and figures
- `analysis_denominator_audit/`: node-normalized loss `L` versus total loss `W`
- `units/*/densify_meta.json`: placement counts and population-fill metadata

Population-weighted and uniform-null scenarios were not included in this artifact.
