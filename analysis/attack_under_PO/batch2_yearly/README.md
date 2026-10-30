# Batch 2 — temporal robustness

With the recommended `EVENT_MODE=replay`, this package runs baseline attacks
under three temporal designs:

| Family | Network | Outage input | Purpose |
| --- | --- | --- | --- |
| A | year `t` | observed events in year `t` | combined network- and outage-year comparison |
| B | fixed 2023 | observed events in year `t` | outage-year sensitivity |
| C | year `t` | pooled 2018–2023 posterior and event pool | network evolution under a fixed outage environment |

## Recommended historical replay for A/B

Set `EVENT_MODE=replay`. In each Monte Carlo replication, every cleaned event
from the selected year is used exactly once. Its county, severity, duration,
and disruption radius remain fixed. Only its within-county, population-weighted
epicenter is redrawn. There is no Bayesian lambda draw, event-count draw,
bootstrap resampling, or fallback to another county or year.

The runner checks that the event identities, counties, durations, radii, and
event count are identical across replications. Use a new output directory so
replay results cannot mix with the legacy lambda-bootstrap results.

```bash
NETWORK_ROOT=$PWD/outputs/1029-network-data-preprocessing-10km-geodesic-2018-2023 \
EVENT_MODE=replay FAMILIES="B" YEARS="2018 2019 2020 2021 2022 2023" \
N_SIMS=100 WORKERS=45 NO_EVENTS=1 \
OUT=$PWD/results_replay_B_1029 \
bash analysis/attack_under_PO/batch2_yearly/run_cluster.sh
```

Outputs include `batch_manifest.json`, `observed_event_counts.csv`, and one
directory per network/outage-year combination. Each unit directory contains
`metrics_baseline.csv`, `summary.csv`, and `manifest.json`.

## Legacy mode

`EVENT_MODE=lambda_bootstrap` preserves the earlier A/B/C implementation. It
draws posterior county rates and resamples historical events. It is retained
only for reproducibility and is not the recommended A/B specification.
