#!/bin/sh
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000

. venv/bin/activate

# DUALFloodGNN
srun python test.py --config 'configs/mswegnn_config.yaml' --model 'DUALFloodGNN' --model_path ''
srun python test.py --config 'configs/mswegnn_no_physics_config.yaml' --model 'DUALFloodGNN' --model_path ''
