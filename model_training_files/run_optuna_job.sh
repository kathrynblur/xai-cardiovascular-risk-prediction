#!/bin/bash
#SBATCH --job-name=cardio_optuna
#SBATCH --output=/home/vault/b192aa/b192aa44/cardio_project/outputs/logs/slurm_%j.out
#SBATCH --error=/home/vault/b192aa/b192aa44/cardio_project/outputs/logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source /etc/profile

# Strict error handling
set -euo pipefail

# Fix unbound LD_LIBRARY_PATH variable warning
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

module purge
module unload python
module load python/3.12-conda

cd /home/vault/b192aa/b192aa44/cardio_project

# Ensure shared utilities are importable
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python /home/vault/b192aa/b192aa44/cardio_project/cardiovascular_optuna_gpu.py