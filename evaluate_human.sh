#!/bin/bash
#SBATCH --job-name=rlcard
#SBATCH --time=300:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=2080rtx|v100|p100
#SBATCH --exclude=linux41,linux42,linux43,linux44,linux45
#SBATCH --partition=compsci-gpu
#SBATCH --output=results/jobs/train_%j.out

source /home/users/shuai/venv/rlcard/bin/activate
hostname

python3 evaluate_human.py