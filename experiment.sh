#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

# Local Weight Search
srun python train.py --config 'configs/local_experiment/local_loss_0.05_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/local_experiment/local_loss_0.01_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.0005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.0001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.00005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.00001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.000005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.000001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.0000005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/local_experiment/local_loss_0.0000001_config.yaml' --model 'NodeEdgeGNN'

# Global Weight Search
# srun python train.py --config 'configs/global_experiment/global_loss_0.005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.0005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.0001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.00005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.00001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.000005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.000001_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.0000005_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/global_experiment/global_loss_0.0000001_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_experiment/global_loss_0.00000005_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_experiment/global_loss_0.00000001_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_experiment/global_loss_0.000000005_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_experiment/global_loss_0.000000001_config.yaml' --model 'NodeEdgeGNN'

# srun python train.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNN'
