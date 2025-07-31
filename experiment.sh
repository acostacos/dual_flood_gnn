#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/experiment_attn/config_stc_no_attn.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/experiment_attn/config_stc_attn.yaml' --model 'NodeEdgeGNNAttn'

# srun python train.py --config 'configs/experiment_auto/config_supervised.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_2_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_3_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_4_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_6_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_8_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_10_step.yaml' --model 'NodeEdgeGNNAttn'
# srun python train.py --config 'configs/experiment_auto/config_autoregressive_12_step.yaml' --model 'NodeEdgeGNNAttn'

# srun python train.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNN'
