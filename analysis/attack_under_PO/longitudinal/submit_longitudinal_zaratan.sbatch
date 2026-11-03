#!/usr/bin/env bash
#SBATCH --job-name=ev-longitudinal
#SBATCH --array=2018-2022
#SBATCH --cpus-per-task=45
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=results_review/logs/longitudinal-%A-%a.out
#SBATCH --error=results_review/logs/longitudinal-%A-%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the EV-Risk-review repository}"
mkdir -p results_review/logs

# Submit from an activated environment, or set VENV_PATH to its directory.
if [[ -n "${VENV_PATH:-}" ]]; then
  export PATH="${VENV_PATH}/bin:${PATH}"
fi

python -c "import arviz, geopandas, networkx, numpy, pandas, scipy"

YEAR="${SLURM_ARRAY_TASK_ID}" \
RUN_DATE="${RUN_DATE:-$(date +%m%d)}" \
N_SIMS="${N_SIMS:-100}" \
WORKERS="${SLURM_CPUS_PER_TASK:-45}" \
bash analysis/attack_under_PO/longitudinal/run_network_year_cluster.sh full
