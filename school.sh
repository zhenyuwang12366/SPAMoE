#!/usr/bin/env bash
#SBATCH --job-name=FWINO_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%j.out
#SBATCH --no-requeue

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

exec bash "${SCRIPT_DIR}/school_local.sh" "$@"
