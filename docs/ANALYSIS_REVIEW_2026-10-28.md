# Manuscript analysis review (2026-10-28)

## Final experiment structure

The paper should use one canonical analysis and two clearly separated checks.

1. **Canonical state comparison.** This remains the main analysis and establishes the
   pooled 2018-2023 state vulnerability ranking. No canonical simulation code or result
   was changed in this review.
2. **Temporal robustness (Batch 2).** This tests whether rankings change when the
   network year and outage year vary separately.
3. **Intervention illustration (Batch 1).** This tests NEVI-style node expansion at
   10%, 20%, 30%, and 50%, with outage-severity reduction as a comparator. It does not
   validate the canonical state ranking.

Population-weighted and uniform placement are optional appendix controls. The urban
radius, 5 km network, within-cluster placement, and duration variants are not needed for
the current manuscript argument and are excluded from the default review run.

## Code checked

The desktop working copy and GitHub `main` were byte-identical for the simulation code
that determines the results:

- `counterfactual_batch1/scenarios.py`
- `counterfactual_batch1/densify_network.py`
- `counterfactual_batch1/crn_engine.py`
- `counterfactual_batch1/run_batch1_cluster.sh`
- `counterfactual_batch1/smoke_test_batch1.py`
- `batch2_yearly/engine.py`
- `batch2_yearly/analyze_batch2.py`
- `batch2_yearly/run_cluster.sh`

The only pre-review desktop/GitHub difference was that the desktop
`analyze_dose_response.load_panel` accepted either `Path` or `str`. That change does not
alter calculations.

### Network distance and efficiency

The graph is spatially embedded: Voronoi/Delaunay construction determines which nearby
hypernodes are connected. The outage analysis then calls NetworkX
`global_efficiency`, which uses unweighted shortest-path hop distance. Stored edge
distance attributes do not enter this calculation. The manuscript should describe the
metric as topological efficiency, not distance-weighted efficiency.

### NEVI-style expansion

For intervention percentage `p`, the code requests

`n_added = ceil(p * n_original)`.

Candidate locations are sampled approximately every 50 miles along interstate
corridors, candidates within 10 km of existing or selected nodes are excluded, and the
largest coverage gaps are filled first. If corridor candidates are insufficient, the
remaining target is filled with population-weighted sites. The correct label is
**NEVI-style interstate corridor expansion**, not strict NEVI compliance.

After adding nodes, the code rebuilds Voronoi/Delaunay adjacency for the full set of old
and new nodes. Added nodes are never selected as outage epicenters, but they can be hit
by an outage footprint.

### Loss denominator

The reported node-normalized loss is

`L = sum(relative efficiency loss * outage duration) / scenario node count`.

Because the denominator grows under node expansion, the intervention analysis must also
report

`W = sum(relative efficiency loss * outage duration)`.

`W` removes the direct node-count denominator but retains changes caused by rebuilt
topology and outage exposure.

## Code correction made in this review

`analyze_dose_response.py` previously injected a baseline dose-0 row and then also kept
the explicit `CF-D_nevi_p0` and `CF-S_m0` rows. CF-D and CF-S slope regressions therefore
used six observations with dose 0 duplicated.

The analyzer now:

- retains an explicit dose-0 row when it exists;
- adds baseline dose 0 only for controls that have no explicit zero-dose scenario;
- rejects duplicate unit-family-dose rows; and
- requires exactly five unique doses for CF-D and CF-S.

After correction, every CF-D and CF-S regression uses five doses. The headline slope
comparison is unchanged: severity reduction is steeper in 26 of 45 units and NEVI-style
expansion is steeper in 19 of 45 units.

## Results retained for review

### Temporal robustness

The imported Batch 2 artifact contains 810 complete unit-year rows:

- A: year-specific network and year-specific outages;
- B: fixed 2023 network and year-specific outages; and
- C: year-specific network and pooled outages.

Each family has 45 units across six years. A-2023 and B-2023 agree exactly for `L_cum`,
`W`, and `P_hit`. There are no missing values, duplicate family-unit-tag keys, or
state-pooled severity fallbacks.

Mean Spearman rank correlations for cumulative loss are 0.579 for A, 0.619 for B, and
0.958 for C. For 2018 versus 2023 they are 0.525, 0.501, and 0.896, respectively. The
ranking is therefore much more stable when outages are pooled than when the outage year
changes. Across the reported year pairs, the outage-instability share (ODI) averages
0.906 for cumulative loss, compared with 0.094 for the network-instability share (NDI).

Allowed conclusion: the pooled canonical ranking describes persistent, long-run
vulnerability, while annual rankings remain sensitive to the realized outage year.
Do not claim that year-specific state rankings are nearly fixed.

The imported artifact does not include the split-half Monte Carlo reliability outputs.
Run the central cluster script in `analyze` mode after the raw Batch 2 metrics are
available.

### Intervention

Across 45 units, mean changes relative to baseline are:

| Added nodes | Node-normalized loss `L` | Total loss `W` |
| ---: | ---: | ---: |
| 10% | -13.62% | -4.50% |
| 20% | -24.29% | -8.72% |
| 30% | -34.41% | -14.41% |
| 50% | -46.43% | -19.48% |

Allowed conclusion: NEVI-style expansion produces a dose-dependent reduction in
node-normalized vulnerability. The reduction in total system loss is smaller, so the
paper must not interpret the full normalized decline as a topology-only mitigation
effect.

Population-weighted and uniform-null placement results are not present in this artifact;
no placement-superiority claim is currently supported.

## Active review entry points

- `analysis/attack_under_PO/run_review_experiments_cluster.sh`
- `analysis/attack_under_PO/counterfactual_batch1/run_batch1_cluster.sh`
- `analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py`
- `analysis/attack_under_PO/counterfactual_batch1/audit_cf_d_denominator.py`
- `analysis/attack_under_PO/batch2_yearly/run_cluster.sh`
- `analysis/attack_under_PO/batch2_yearly/analyze_batch2.py`
- `analysis/attack_under_PO/batch2_yearly/check_mc_reliability.py`

Historical scripts remain available for provenance, but they are not part of the default
manuscript review workflow.

Run the compact artifact check with:

```bash
python tests/validate_review_artifacts.py
```

## Artifact locations

- `artifacts/cf_batch1_2023_pooled/`
- `artifacts/batch2_yearly/analysis_batch2/`
