#!/bin/sh
#SBATCH --job-name=hyperparameter_search
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

. venv/bin/activate

# Global Mass Loss hyperparameter search
#srun python hyperparameter_search.py --config 'configs/hp_search_global_config.yaml' --model 'GAT'
srun python hyperparameter_search.py  --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'global_mass_loss' --summary_file 'temp_train.csv'

# Local Mass Loss hyperparameter search
srun python hyperparameter_search.py --config 'configs/hp_search_local_config.yaml' --model 'GAT'

# Global and Local Mass Loss hyperparameter search
srun python hyperparameter_search.py --config 'configs/hp_search_config.yaml' --model 'GAT'

# Base NodeEdge GNN hyperparameter search
#srun python hyperparameter_search.py --config 'configs/hp_search_no_physics_config.yaml' --model 'NodeEdgeGNN'
srun python hyperparameter_search.py  --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNN' --hyperparameters 'edge_pred_loss' --summary_file 'temp_train.csv'
