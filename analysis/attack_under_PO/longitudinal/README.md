# Longitudinal network-year analysis

This runner holds the pooled 2018--2023 outage posterior, historical event
pool, radius model, and Monte Carlo seeds fixed while changing only the
charging-network year. This isolates changes associated with network evolution.

The 2023 network is the canonical/reference-year analysis. The longitudinal
extension runs 2018--2022 with the same 100 one-year Monte Carlo replications.
Each year writes to a separate result directory and can run on a separate
Zaratan allocation.

Submit from the repository root after activating the Python environment:

```bash
mkdir -p results_review/logs
sbatch analysis/attack_under_PO/longitudinal/submit_longitudinal_zaratan.sbatch
```

Alternatively, set `VENV_PATH` when submitting if the environment is not
already active in the Slurm job.
