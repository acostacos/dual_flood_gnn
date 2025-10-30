#!/bin/sh
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000

. venv/bin/activate

# DUALFloodGNN
# Base Architecture
srun python test.py --config 'configs/config.yaml' --model 'DUALFloodGNN' --model_path ''

# Physics-Informed Variants
srun python test.py --config 'configs/global_loss_config.yaml' --model 'DUALFloodGNN' --model_path ''
srun python test.py --config 'configs/local_loss_config.yaml' --model 'DUALFloodGNN' --model_path ''
srun python test.py --config 'configs/no_physics_config.yaml' --model 'DUALFloodGNN' --model_path ''

# Standard GNN Architectures
# Node Prediction
srun python test.py --config 'configs/standard_gnn_config.yaml' --model 'GAT' --model_path ''
srun python test.py --config 'configs/standard_gnn_config.yaml' --model 'GCN' --model_path ''

# Edge Prediction
srun python test.py --config 'configs/standard_gnn_config.yaml' --model 'EdgeGAT' --model_path ''
srun python test.py --config 'configs/standard_gnn_config.yaml' --model 'EdgeGCN' --model_path ''
