#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/experiment/config_7200.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_3600.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_1200.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_600.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_300.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_30.yaml' --model 'NodeEdgeGNN' --with_test True
