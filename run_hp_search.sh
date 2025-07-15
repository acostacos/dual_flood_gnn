#!/bin/sh
#SBATCH --job-name=hyperparameter_search
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

# Grid Search
# Global Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'global_mass_loss' --summary_file 'temp_train.csv'

# Local Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'local_mass_loss' --summary_file 'temp_train.csv'

# Global and Local Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'global_mass_loss' 'local_mass_loss' --summary_file 'temp_train.csv'

# Base NodeEdge GNN hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNN' --hyperparameters 'edge_pred_loss' --summary_file 'temp_train.csv'


# Bayesian Search
# Global Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'global_mass_loss' --summary_file 'temp_train.csv'

# Local Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'local_mass_loss' --summary_file 'temp_train.csv'

# Global and Local Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'GAT' --hyperparameters 'global_mass_loss' 'local_mass_loss' --summary_file 'temp_train.csv'

# Base NodeEdge GNN hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNN' --hyperparameters 'edge_pred_loss' --summary_file 'temp_train.csv'
