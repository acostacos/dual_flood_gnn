#!/bin/sh
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000
#SBATCH --time=720

. venv/bin/activate

srun python train.py --config 'configs/config.yaml' --model 'GAT'
srun python train.py --config 'configs/config.yaml' --model 'GCN'
