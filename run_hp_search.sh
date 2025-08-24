#!/bin/sh
#SBATCH --job-name=hyperparameter_search
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

. venv/bin/activate

# Grid Search
# Base NodeEdge GNN hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' --summary_file 'train.csv'

# Global Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'global_mass_loss' --summary_file 'train.csv'

# Local Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'local_mass_loss' --summary_file 'train.csv'

# Global and Local Mass Loss hyperparameter search
srun python grid_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'global_mass_loss' 'local_mass_loss' --summary_file 'train.csv'


# Bayesian Search
# Base NodeEdge GNN hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' --summary_file 'train.csv'

# Global Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'global_mass_loss' --summary_file 'train.csv'

# Local Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'local_mass_loss' --summary_file 'train.csv'

# Global and Local Mass Loss hyperparameter search
srun python bayesian_search.py --config 'configs/hp_search_config.yaml' --model 'NodeEdgeGNNAttn' --hyperparameters 'edge_pred_loss' 'global_mass_loss' 'local_mass_loss' --summary_file 'train.csv'
