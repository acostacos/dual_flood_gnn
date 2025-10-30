#!/bin/sh
#SBATCH --job-name=hyperparameter_search
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

# Base GNN hyperparameter search
srun python hp_search.py --config 'configs/no_physics_config.yaml' --hparam_config 'configs/hparam_config/overall_model_config.yaml' --model 'DUALFloodGNN'
srun python hp_search.py --config 'configs/no_physics_config.yaml' --hparam_config 'configs/hparam_config/edge_pred_loss.yaml' --model 'DUALFloodGNN'

# Global Mass Conservation Loss hyperparameter search
srun python hp_search.py --config 'configs/global_loss_config.yaml' --hparam_config 'configs/hparam_config/global_mass_loss.yaml' --model 'DUALFloodGNN'

# Local Mass Conservation Loss hyperparameter search
srun python hp_search.py --config 'configs/local_loss_config.yaml' --hparam_config 'configs/hparam_config/local_mass_loss.yaml' --model 'DUALFloodGNN'

# Global and Local Mass Conservation Loss hyperparameter search
srun python hp_search.py --config 'configs/config.yaml' --hparam_config 'configs/hparam_config/global_and_local_mass_loss.yaml' --model 'DUALFloodGNN'
