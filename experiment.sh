#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNN'
