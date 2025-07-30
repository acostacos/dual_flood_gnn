#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/experiment_num_layers/config_1layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_2layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_3layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_4layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_6layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_8layer.yaml' --model 'NodeEdgeGNNAttn'
srun python train.py --config 'configs/experiment_num_layers/config_10layer.yaml' --model 'NodeEdgeGNNAttn'

# srun python train.py --config 'configs/experiment_auto/config_supervised.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_2_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_3_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_4_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_6_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_8_step.yaml' --model 'NodeEdgeGNNAttn'

# srun python train.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNNAttn'
