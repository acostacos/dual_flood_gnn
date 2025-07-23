#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/experiment/config_supervised.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_autoregressive_2_step.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_autoregressive_3_step.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_autoregressive_4_step.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_autoregressive_6_step.yaml' --model 'NodeEdgeGNN' --with_test True
srun python train.py --config 'configs/experiment/config_autoregressive_8_step.yaml' --model 'NodeEdgeGNN' --with_test True
