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
  realized by the corresponding corridor scenario, but prioritizes the most
  populated census tracts under the same 10 km separation rule. This isolates
  placement strategy from expansion size.
- **Severity reduction (`CF-S`)** reduces affected customers by the same doses,
  then reapplies coverage correction, the county-customer cap, and the radius
  calculation.

Run the input check, a two-simulation smoke test, and then the full experiment:

```bash
python -u analysis/attack_under_PO/counterfactual_batch1/download_ev_afc.py

python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py --check-only

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

The primary comparison measures unweighted global efficiency only among the
original 2023 hypernodes. Added nodes may connect paths, but are not counted as
endpoints or in the denominator. The all-node metric is retained as a secondary
output.
