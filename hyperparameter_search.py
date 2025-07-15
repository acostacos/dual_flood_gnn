import os
import numpy as np
import traceback
import torch
import optuna
import pandas as pd
import shutil

from argparse import ArgumentParser, Namespace
from data import FloodEventDataset, InMemoryFloodEventDataset
from models import model_factory
from models.base_model import BaseModel
from optuna.visualization import plot_optimization_history, plot_slice
from test import test_autoregressive, test_autoregressive_node_only
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from training import NodeRegressionTrainer, DualRegressionTrainer
from typing import List, Optional, Tuple
from utils import ValidationStats, Logger, file_utils

TEMP_DIR_NAME = '_temp_hp_dir'
HYPERPARAMETER_CHOICES = [
    'global_mass_loss',
    'local_mass_loss',
    'edge_pred_loss',
]
dataset_cache = {}

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--hyperparameters", type=str, choices=HYPERPARAMETER_CHOICES, nargs='+', required=True, help='Hyperparameters to search for')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--summary_file", type=str, required=True, help='Dataset summary file for hyperparameter search. Events in file will be used for cross-validation')
    parser.add_argument("--num_trials", type=int, default=20, help='Number of trials for hyperparameter search')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    return parser.parse_args()

def get_hyperparam_search_config() -> Tuple[bool, bool, bool]:
    hyperparams_to_search = args.hyperparameters

    use_global_mass_loss = 'global_mass_loss' in hyperparams_to_search
    use_local_mass_loss = 'local_mass_loss' in hyperparams_to_search
    use_edge_pred_loss = 'edge_pred_loss' in hyperparams_to_search
    return use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss

def create_cross_val_dataset_files() -> List[str]:
    dataset_summary_file = args.summary_file

    dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

    hec_ras_run_ids = []
    for _, row in summary_df.iterrows():
        run_id = row['Run_ID']
        assert run_id not in hec_ras_run_ids, f'Duplicate Run_ID found: {run_id}'

        other_rows_df: pd.DataFrame = summary_df[summary_df['Run_ID'] != run_id]
        train_df_path = os.path.join(raw_temp_dir_path, f'train_{run_id}.csv') 
        other_rows_df.to_csv(train_df_path, index=False)

        event_df = pd.DataFrame([row])
        test_df_path = os.path.join(raw_temp_dir_path, f'test_{run_id}.csv')
        event_df.to_csv(test_df_path, index=False)

        hec_ras_run_ids.append(run_id)

    return hec_ras_run_ids

def load_datasets(run_id: str) -> Tuple[FloodEventDataset, FloodEventDataset]:
    if f'train_{run_id}' in dataset_cache and f'test_{run_id}' in dataset_cache:
        return dataset_cache[f'train_{run_id}'], dataset_cache[f'test_{run_id}']

    features_stats_file = os.path.join(TEMP_DIR_NAME, f'features_stats_{run_id}.yaml')
    train_dataset_summary_file = os.path.join(TEMP_DIR_NAME, f'train_{run_id}.csv')
    train_event_stats_file = os.path.join(TEMP_DIR_NAME, f'train_event_stats_{run_id}.yaml')
    train_dataset_config = {
        **base_dataset_config,
        'mode': 'train',
        'dataset_summary_file': train_dataset_summary_file,
        'event_stats_file': train_event_stats_file,
        'features_stats_file': features_stats_file,
        'with_global_mass_loss': use_global_mass_loss,
        'with_local_mass_loss': use_local_mass_loss,
    }

    test_dataset_summary_file = os.path.join(TEMP_DIR_NAME, f'test_{run_id}.csv')
    test_event_stats_file = os.path.join(TEMP_DIR_NAME, f'test_event_stats_{run_id}.yaml')
    test_dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': test_dataset_summary_file,
        'event_stats_file': test_event_stats_file,
        'features_stats_file': features_stats_file,
        # Exclude computation of physics loss for hyperparameter search
        'with_global_mass_loss': False,
        'with_local_mass_loss': False,
    }

    storage_mode = dataset_parameters['storage_mode']
    dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset

    train_dataset = dataset_class(**train_dataset_config)
    dataset_cache[f'train_{run_id}'] = train_dataset

    test_dataset = dataset_class(**test_dataset_config)
    dataset_cache[f'test_{run_id}'] = test_dataset

    return train_dataset, test_dataset

def load_model(dataset: FloodEventDataset) -> BaseModel:
    base_model_params = {
        'static_node_features': dataset.num_static_node_features,
        'dynamic_node_features': dataset.num_dynamic_node_features,
        'static_edge_features': dataset.num_static_edge_features,
        'dynamic_edge_features': dataset.num_dynamic_edge_features,
        'previous_timesteps': dataset.previous_timesteps,
        'device': args.device,
    }
    model_config = {**model_params, **base_model_params}
    model = model_factory(args.model, **model_config)
    return model

def cross_validate(global_mass_loss_percent: Optional[float],
                   local_mass_loss_percent: Optional[float],
                   edge_pred_loss_percent: Optional[float]) -> float | Tuple[float, float]:
    val_rmses = []
    if use_edge_pred_loss:
        val_edge_rmses = []
    for run_id in hec_ras_run_ids:
        logger.log(f'Cross-validating with Run ID {run_id} as the test set...\n')

        train_dataset, test_dataset = load_datasets(run_id)
        model = load_model(train_dataset)
        delta_t = train_dataset.timestep_interval

        # ============ Training Phase ============
        train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'])
        criterion = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

        if global_mass_loss_percent is None:
            global_mass_loss_percent = config['loss_func_parameters']['global_mass_loss_percent']
        if local_mass_loss_percent is None:
            local_mass_loss_percent = config['loss_func_parameters']['local_mass_loss_percent']
        if edge_pred_loss_percent is None:
            edge_pred_loss_percent = config['loss_func_parameters']['edge_pred_loss_percent']
        trainer_params = {
            'model': model,
            'dataloader': train_dataloader,
            'optimizer': optimizer,
            'loss_func': criterion,
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_percent': global_mass_loss_percent,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_percent': local_mass_loss_percent,
            'delta_t': delta_t,
            'num_epochs': train_config['num_epochs'],
            'logger': logger,
            'device': args.device,
        }
        if use_edge_pred_loss:
            trainer = DualRegressionTrainer(**trainer_params, edge_pred_loss_percent=edge_pred_loss_percent)
        else:
            trainer = NodeRegressionTrainer(**trainer_params)

        trainer.training_stats.log = lambda x : None  # Suppress training stats logging to console
        trainer.train()
        trainer.training_stats.log = logger.log  # Restore logging to console

        trainer.print_stats_summary()

        # ============ Testing Phase ============
        logger.log('\nValidating model...')

        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        validation_stats = ValidationStats(logger=logger)
        if use_edge_pred_loss:
            test_autoregressive(model, test_dataset, 0, validation_stats, rollout_start, rollout_timesteps, args.device, include_physics_loss=False)
        else:
            test_autoregressive_node_only(model, test_dataset, 0, validation_stats, rollout_start, rollout_timesteps, args.device, include_physics_loss=False)

        avg_rmse = validation_stats.get_avg_rmse()
        val_rmses.append(avg_rmse)
        logger.log(f'Event {run_id} RMSE: {avg_rmse:.4e}')

        if use_edge_pred_loss:
            avg_edge_rmse = validation_stats.get_avg_edge_rmse()
            val_edge_rmses.append(avg_edge_rmse)
            logger.log(f'Event {run_id} Edge RMSE: {avg_edge_rmse:.4e}')

    def get_avg_rmse(rmses: List[float]) -> float:
        np_rmses = np.array(rmses)
        is_finite = np.isfinite(np_rmses)
        if np.any(is_finite):
            return np_rmses[is_finite].mean()
        return 1e10

    avg_val_rmse = get_avg_rmse(val_rmses)
    logger.log(f'\nAverage RMSE across all events: {avg_val_rmse:.4e}')
    if not use_edge_pred_loss:
        return avg_val_rmse

    avg_val_edge_rmse = get_avg_rmse(val_edge_rmses)
    logger.log(f'Average Edge RMSE across all events: {avg_val_edge_rmse:.4e}')
    return avg_val_rmse, avg_val_edge_rmse

def objective(trial: optuna.Trial) -> float:
    global_mass_loss_percent = trial.suggest_float('global_mass_loss_percent', 0.001, 0.5, log=True) if use_global_mass_loss else None
    local_mass_loss_percent = trial.suggest_float('local_mass_loss_percent', 0.001, 0.5, log=True) if use_local_mass_loss else None
    edge_pred_loss_percent = trial.suggest_float('edge_pred_loss_percent', 0.1, 0.9, step=0.05) if use_edge_pred_loss else None

    logger.log(f'Hyperparameters: global_mass_loss_percent={global_mass_loss_percent}, local_mass_loss_percent={local_mass_loss_percent}, edge_pred_loss_percent={edge_pred_loss_percent}')

    return cross_validate(global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent)

def plot_hyperparameter_search_results(study: optuna.Study):
    stats_dir = train_config['stats_dir']
    if stats_dir is None:
        return

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    if use_edge_pred_loss:
        fig = plot_optimization_history(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, 'optimization_history.html'))

        fig = plot_optimization_history(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, 'edge_optimization_history.html'))

        fig = plot_slice(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, 'slice_plot.html'))

        fig = plot_slice(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, 'edge_slice_plot.html'))
    else:
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(stats_dir, 'optimization_history.html'), target_name='RMSE')

        fig = plot_slice(study)
        fig.write_html(os.path.join(stats_dir, 'slice_plot.html'), target_name='RMSE')

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    # Initialize logger
    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss = get_hyperparam_search_config()
        assert use_global_mass_loss or use_local_mass_loss or use_edge_pred_loss, 'At least one hyperparameter must be selected for search'
        assert not use_edge_pred_loss or 'NodeEdgeGNN' in args.model, 'Edge prediction loss can only be used with NodeEdgeGNN model'

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Print static configuration
        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # ============ Dataset Configuration ============
        dataset_parameters = config['dataset_parameters']
        root_dir = dataset_parameters['root_dir']
        base_dataset_config = {
            'root_dir': root_dir,
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
            'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
            'logger': logger,
            'force_reload': False,
        }
        logger.log(f'Using train dataset configuration: {base_dataset_config}')

        # ============ Model Configuration ============
        model_params = config['model_parameters'][args.model]
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_params}')

        # ============ Training Configuration ============
        train_config = config['training_parameters']
        logger.log(f'Using training configuration: {train_config}')

        # ============ Testing Configuration ============
        test_config = config['testing_parameters']
        logger.log(f'Using testing configuration: {test_config}')

        # Begin hyperparameter search
        # Create temporary directories
        raw_temp_dir_path = os.path.join(root_dir, 'raw', TEMP_DIR_NAME)
        processed_temp_dir_path = os.path.join(root_dir, 'processed', TEMP_DIR_NAME)
        if os.path.exists(processed_temp_dir_path):
            shutil.rmtree(processed_temp_dir_path)
        if os.path.exists(raw_temp_dir_path):
            shutil.rmtree(raw_temp_dir_path)
        os.makedirs(raw_temp_dir_path)

        hec_ras_run_ids = create_cross_val_dataset_files()

        study_kwargs = {}
        if use_edge_pred_loss:
            study_kwargs['directions'] = ['minimize', 'minimize']
        else:
            study_kwargs['direction'] = 'minimize'

        study = optuna.create_study(**study_kwargs)
        logger.log(f'Using sampler: {study.sampler.__class__.__name__ if study.sampler else None}')
        logger.log(f'Using pruner: {study.pruner.__class__.__name__ if study.pruner else None}')
        logger.log(f'Running hyperparameter search for {args.num_trials} trials...')
        study.optimize(objective, n_trials=args.num_trials)

        if use_edge_pred_loss:
            logger.log('Best hyperparameters found:')
            for trial in study.best_trials:
                logger.log(f'Trial {trial.number}:')
                for key, value in trial.params.items():
                    logger.log(f'\t{key}: {value}')
                objective_values_str = ', '.join([f'{v:.4e}' for v in trial.values])
                logger.log(f'\tObjective values: {objective_values_str}')
        else:
            logger.log('Best hyperparameters found:')
            for key, value in study.best_params.items():
                logger.log(f'{key}: {value}')
            logger.log(f'Best objective value: {study.best_value:.4e}')

        # Plot hyperparameter search results
        plot_hyperparameter_search_results(study)

        # Clean up temporary directories
        shutil.rmtree(raw_temp_dir_path)
        shutil.rmtree(processed_temp_dir_path)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')
