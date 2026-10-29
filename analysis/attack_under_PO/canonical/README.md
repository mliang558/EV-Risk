# Canonical 2023 attack

This folder is the single entry point for the manuscript's canonical attack.
It reuses the 2023 network, the pooled 2018--2023 county posterior, and the
pooled cleaned historical event library. It does not run the counterfactual
or robustness scenarios.

The simulation chain is locked as follows:

1. Draw one joint posterior realization of every county rate `lambda_c`.
2. For each state, draw the annual total from `Poisson(sum_c lambda_c)`.
3. Conditional on that total, select counties with probability proportional
   to `lambda_c`.
4. Bootstrap severity and duration from the selected county's historical
   events, with replacement.
5. Draw an epicenter within that county, calculate the disruption radius,
   remove nodes within the radius, and evaluate unweighted global efficiency.
6. Reset the network before evaluating the next event.

Post-attack efficiency retains the original network's `N(N-1)` denominator;
removed-node and disconnected pairs contribute zero. This prevents node
removal from appearing to improve performance merely by shrinking the set of
pairs being averaged.

Run `run_canonical_cluster.sh check` before the full simulation and
`run_canonical_cluster.sh validate` afterward. The formal output root is
`results_canonical_2023_pooled_sumlambda/`.

The county-selection construction preserves each county's marginal Poisson
rate. It does not preserve temporal ordering or multi-county storm dependence;
the canonical loss is therefore an annual sum of independently evaluated
county-event stress tests, not a chronological cascading-outage simulation.
