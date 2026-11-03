# 2023 intervention experiments

This package compares interventions on the validated 2023
network. Every Monte Carlo replication uses the same pooled 2018–2023 outage
draw across the baseline and all intervention doses.

- **Corridor expansion (`CF-D`)** adds hypernodes along FHWA-designated EV
  Alternative Fuel Corridors at target doses of 10%, 20%, 30%, and 50%.
  Contiguous corridor segments are merged by state and corridor ID, candidate
  points are sampled every 50 miles, and selected sites must remain at least
  10 km from existing and newly added nodes.
  The complete Voronoi network is rebuilt after placement. Corridor shortfalls
  are reported and are never filled with off-corridor sites.
- **Demand-targeted expansion (`Demand`)** adds exactly the number of nodes
  realized by the corresponding corridor scenario. Tracts are ordered by total
  population, and each tract contributes its nearest Census TIGER/Line primary-
  road point to the tract's internal representative point. Sites must lie on a
  primary road inside the tract and remain at least 10 km from original and
  newly added hypernodes. If one candidate per tract cannot match the corridor
  count, the same rule for every state samples on-road candidates first at
  5 km and then, only if needed, at 1 km resolution. Candidate resolution does
  not relax the final 10 km separation. A run fails rather than silently
  accepting a node-count shortfall. This isolates placement strategy from
  expansion size.
- **Severity reduction (`CF-S`)** reduces affected customers by the same doses,
  then reapplies coverage correction, the county-customer cap, and the radius
  calculation.

Outage epicenters are sampled uniformly within the selected county polygon and
are shared across the baseline and every intervention scenario. Census-tract
population affects Demand placement only; it does not affect outage locations.

Run the input check, a two-simulation smoke test, and then the full experiment:

```bash
python -u analysis/attack_under_PO/build_pop_units_epicenter.py \
  --pop-csv analysis/attack_under_PO/counterfactual_batch1/inputs/tract_population_2020.csv.gz \
  --out data/processed/tract_population_units_2020.gpkg

python -u analysis/attack_under_PO/counterfactual_batch1/download_ev_afc.py
python -u analysis/attack_under_PO/counterfactual_batch1/download_primary_roads.py

python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --families baseline,Demand --epicenter-mode county_uniform --check-only

python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --only VT --n-sims 2 --workers 1 \
  --out results_interventions_1031_smoke_VT

python -u analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
  --batch-dir results_interventions_1031_smoke_VT --unit VT

nohup python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --out results_interventions_1031_network2023_pooled2018_2023_n100 \
  --n-sims 100 --workers 45 --no-events \
  > results_review/logs/interventions_1029.log 2>&1 &
```

Analyze the completed run with:

```bash
python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_interventions_1031_network2023_pooled2018_2023_n100
```

To rerun only the corrected Demand comparator, retain the baseline so that the
shared-random-number difference remains directly estimable:

```bash
nohup python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --families baseline,Demand \
  --epicenter-mode county_uniform \
  --population-units data/processed/tract_population_units_2020.gpkg \
  --densify-cache outputs/1102-demand-expansion-2023-pop2020 \
  --out results_demand_1102_network2023_pooled2018_2023_n100 \
  --n-sims 100 --workers 45 --no-events \
  > results_review/logs/demand_1102.log 2>&1 &

python -u analysis/attack_under_PO/counterfactual_batch1/validate_demand_outputs.py \
  --out results_demand_1102_network2023_pooled2018_2023_n100 \
  --expected-sims 100
```

The primary comparison measures unweighted global efficiency only among the
original 2023 hypernodes. Added nodes may connect paths, but are not counted as
endpoints or in the denominator. The all-node metric is retained as a secondary
output.
