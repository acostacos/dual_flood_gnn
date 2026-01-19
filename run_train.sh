#!/bin/sh
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

. venv/bin/activate

# DUALFloodGNN
srun python train.py --config 'configs/mswegnn_config.yaml' --model 'DUALFloodGNN' --seed 666
srun python train.py --config 'configs/mswegnn_no_physics_config.yaml' --model 'DUALFloodGNN' --seed 666
