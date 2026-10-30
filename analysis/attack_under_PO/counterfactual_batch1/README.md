# 2023 intervention experiments

This package compares two one-at-a-time interventions on the validated 2023
network. Every Monte Carlo replication uses the same pooled 2018–2023 outage
draw across the baseline and all intervention doses.

- **Corridor expansion (`CF-D`)** adds eligible Interstate-corridor hypernodes
  at target doses of 10%, 20%, 30%, and 50%. Candidates are sampled every 50
  miles and must remain at least 10 km from existing and newly added nodes.
  The complete Voronoi network is rebuilt after placement. Corridor shortfalls
  are reported and are never filled with off-corridor sites.
- **Severity reduction (`CF-S`)** reduces affected customers by the same doses,
  then reapplies coverage correction, the county-customer cap, and the radius
  calculation.

Run the input check, a two-simulation smoke test, and then the full experiment:

```bash
python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py --check-only

python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --only VT --n-sims 2 --workers 1 \
  --out results_interventions_1029_smoke_VT

python -u analysis/attack_under_PO/counterfactual_batch1/smoke_test_batch1.py \
  --batch-dir results_interventions_1029_smoke_VT --unit VT

nohup python -u analysis/attack_under_PO/counterfactual_batch1/run_batch1_parallel.py \
  --out results_interventions_1029_network2023_pooled2018_2023_n100 \
  --n-sims 100 --workers 45 --no-events \
  > results_review/logs/interventions_1029.log 2>&1 &
```

Analyze the completed run with:

```bash
python analysis/attack_under_PO/counterfactual_batch1/analyze_dose_response.py \
  --batch-dir results_interventions_1029_network2023_pooled2018_2023_n100
```

The analysis reports both the intervention curves and unit-level slopes. The
network metric remains unweighted global efficiency with the canonical fixed
original-node denominator after each independent event.
