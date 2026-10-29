#!/usr/bin/env bash
#SBATCH --job-name=ev-canonical
#SBATCH --cpus-per-task=45
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=results_review/logs/canonical-%j.out
#SBATCH --error=results_review/logs/canonical-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the EV-Risk-review repository}"
export PATH="/opt/data_repo/mliang_work/venv_cad/bin:${PATH}"

N_SIMS="${N_SIMS:-100}" \
WORKERS="${SLURM_CPUS_PER_TASK:-45}" \
bash analysis/attack_under_PO/canonical/run_canonical_cluster.sh full
