#!/bin/sh
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000

. venv/bin/activate

srun python test.py --config 'configs/config.yaml' --model 'GAT' --model_path ''
srun python test.py --config 'configs/config.yaml' --model 'GCN' --model_path ''
