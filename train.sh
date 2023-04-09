#!/bin/bash
#SBATCH --job-name=rlcard
#SBATCH --time=300:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --constraint=2080rtx|v100|p100
#SBATCH --exclude=linux41,linux42,linux43,linux44,linux45
#SBATCH --partition=compsci-gpu
#SBATCH --output=results/jobs/train_%j.out

source /home/users/shuai/venv/rlcard/bin/activate
hostname

# train by 2-gpu
#python3 train_dmc.py --env swy-blm --xpid swy-blm --cuda 0,1 --num_actor_devices 1 --training_device 0 --num_actors 8 --savedir results/dmc_results --total_iterations 2000000

# train by cpu only
python3 train_dmc.py --env swy-blm  --xpid swy-blm --num_actor_devices 1 --training_device 0 --num_actors 8 --savedir results/dmc_results_cpu --total_iterations 2000000
