#!/bin/sh
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

. venv/bin/activate

# DUALFloodGNN
# Base Architecture
srun python train.py --config 'configs/config.yaml' --model 'DUALFloodGNN'

# Physics-Informed Variants
srun python train.py --config 'configs/global_loss_config.yaml' --model 'DUALFloodGNN'
srun python train.py --config 'configs/local_loss_config.yaml' --model 'DUALFloodGNN'
srun python train.py --config 'configs/no_physics_config.yaml' --model 'DUALFloodGNN'

# Standard GNN Architectures
# Node Prediction
srun python train.py --config 'configs/standard_gnn_config.yaml' --model 'GAT'
srun python train.py --config 'configs/standard_gnn_config.yaml' --model 'GCN'

# Edge Prediction
srun python train.py --config 'configs/standard_gnn_config.yaml' --model 'EdgeGAT'
srun python train.py --config 'configs/standard_gnn_config.yaml' --model 'EdgeGCN'
