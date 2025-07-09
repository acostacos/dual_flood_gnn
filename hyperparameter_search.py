import os
import numpy as np
import traceback
import torch

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset
from itertools import product
from models import model_factory
from test import get_test_dataset_config, test_autoregressive, test_autoregressive_node_only
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from training import NodeRegressionTrainer, DualRegressionTrainer
from typing import Dict, Tuple
from utils import ValidationStats, Logger, file_utils

torch.serialization.add_safe_globals([datetime])

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def load_datasets(config: Dict, logger: Logger, debug: bool = False) -> Tuple[FloodEventDataset, FloodEventDataset]:
    dataset_parameters = config['dataset_parameters']
    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'nodes_shp_file': dataset_parameters['nodes_shp_file'],
        'edges_shp_file': dataset_parameters['edges_shp_file'],
        'features_stats_file': dataset_parameters['features_stats_file'],
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
        'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
        'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
        'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        'debug': debug,
        'logger': logger,
        'force_reload': True,
    }
    storage_mode = dataset_parameters['storage_mode']
    dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset

    train_dataset_parameters = dataset_parameters['training']
    train_dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    train_event_stats_file = train_dataset_parameters['event_stats_file']
    loss_func_parameters = config['loss_func_parameters']
    use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
    use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
    train_dataset_config = {
        **base_dataset_config,
        'mode': 'train',
        'dataset_summary_file': train_dataset_summary_file,
        'event_stats_file': train_event_stats_file,
        'with_global_mass_loss': use_global_mass_loss,
        'with_local_mass_loss': use_local_mass_loss,
    }
    logger.log(f'Using train dataset configuration: {train_dataset_config}')
    train_dataset = dataset_class(**train_dataset_config)
    logger.log(f'Loaded train dataset with {len(train_dataset)} samples')

    test_dataset_config = get_test_dataset_config(base_dataset_config, config)
    logger.log(f'Using test dataset configuration: {test_dataset_config}')
    test_dataset = dataset_class(**test_dataset_config)
    logger.log(f'Loaded test dataset with {len(train_dataset)} samples')

    return train_dataset, test_dataset

def get_hyperparameters_for_search(hyperparam_comb: Tuple,  loss_func_parameters: Dict, model_name: str) -> Dict:
    index = 0

    use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
    global_mass_loss_percent = loss_func_parameters['global_mass_loss_percent']
    if use_global_mass_loss:
        global_mass_loss_percent = hyperparam_comb[index]
        index += 1

    use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
    local_mass_loss_percent = loss_func_parameters['local_mass_loss_percent']
    if use_local_mass_loss:
        local_mass_loss_percent = hyperparam_comb[index]
        index += 1

    use_edge_pred_loss = model_name == 'NodeEdgeGNN'
    edge_pred_loss_percent = loss_func_parameters['edge_pred_loss_percent']
    if use_edge_pred_loss:
        edge_pred_loss_percent = hyperparam_comb[index]
        index += 1
    return global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']

    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Check which hyperparameters will be searched
        loss_func_parameters = config['loss_func_parameters']
        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
        use_edge_pred_loss = args.model == 'NodeEdgeGNN'
        if not (use_global_mass_loss or use_local_mass_loss or use_edge_pred_loss):
            raise ValueError('No hyperparameters to search. Please enable at least one of the mass loss functions or edge prediction loss.')

        best_hyperparameters = {}
        HYPERPARAMETERS = {}
        if use_global_mass_loss:
            GLOBAL_LOSS_PERCENTS = [0.01, 0.001, 0.0001, 0.00001]
            HYPERPARAMETERS['global_mass_loss_percent'] = GLOBAL_LOSS_PERCENTS
            best_hyperparameters['global_mass_loss_percent'] = None
        if use_local_mass_loss:
            LOCAL_LOSS_PERCENTS = [0.01, 0.001, 0.0001, 0.00001]
            HYPERPARAMETERS['local_mass_loss_percent'] = LOCAL_LOSS_PERCENTS
            best_hyperparameters['local_mass_loss_percent'] = None
        if use_edge_pred_loss:
            EDGE_LOSS_PERCENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
            HYPERPARAMETERS['edge_pred_loss_percent'] = EDGE_LOSS_PERCENTS
            best_hyperparameters['edge_pred_loss_percent'] = None

        hyperparameter_list = list(HYPERPARAMETERS.keys())
        logger.log(f'Performing hyperparameter search for the following hyperparameters: {', '.join(hyperparameter_list)}')

        # Load datasets
        train_dataset, test_dataset = load_datasets(config, logger, args.debug)

        # Load model configuration
        model_params = config['model_parameters'][args.model]
        base_model_params = {
            'static_node_features': train_dataset.num_static_node_features,
            'dynamic_node_features': train_dataset.num_dynamic_node_features,
            'static_edge_features': train_dataset.num_static_edge_features,
            'dynamic_edge_features': train_dataset.num_dynamic_edge_features,
            'previous_timesteps': train_dataset.previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')

        train_config = config['training_parameters']
        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        num_epochs = train_config['num_epochs']

        log_train_config = {'num_epochs': num_epochs, 'batch_size': train_config['batch_size'], 'learning_rate': train_config['learning_rate'], 'weight_decay': train_config['weight_decay'] }
        logger.log(f'Using training configuration: {log_train_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        log_test_config = {'rollout_start': rollout_start, 'rollout_timesteps': rollout_timesteps}
        logger.log(f'Using testing configuration: {log_test_config}')

        delta_t = train_dataset.timestep_interval

        best_rmse = float('inf')
        hyperparameter_values = [HYPERPARAMETERS[key] for key in hyperparameter_list]
        combinations = list(product(*hyperparameter_values))
        for comb in combinations:
            logger.log('\nRunning training with hyperparameter combination:')
            for key, value in zip(hyperparameter_list, comb):
                logger.log(f'\t{key}: {value}')

            search_hyperparams = get_hyperparameters_for_search(comb, loss_func_parameters, args.model)
            global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent = search_hyperparams

            # ============ Training Phase ============
            model = model_factory(args.model, **model_config)
            train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'])
            criterion = MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

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
                'num_epochs': num_epochs,
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

            # Save training stats and model
            curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if stats_dir is not None:
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)

                saved_metrics_path = os.path.join(stats_dir, f'{args.model}_{curr_date_str}_train_stats.npz')
                trainer.save_stats(saved_metrics_path)

            if model_dir is not None:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_path = os.path.join(model_dir, f'{args.model}_{curr_date_str}.pt')
                trainer.save_model(model_path)

            # ============ Testing Phase ============
            logger.log('\nValidating model on test dataset...')

            events_rmse = []
            for event_idx, run_id in enumerate(test_dataset.hec_ras_run_ids):
                validation_stats = ValidationStats(logger=logger)
                if use_edge_pred_loss:
                    test_autoregressive(model, test_dataset, event_idx, validation_stats, rollout_start, rollout_timesteps, args.device)
                else:
                    test_autoregressive_node_only(model, test_dataset, event_idx, validation_stats, rollout_start, rollout_timesteps, args.device)
                avg_rmse = validation_stats.get_avg_rmse()
                events_rmse.append(avg_rmse)
                logger.log(f'Event {run_id} RMSE: {avg_rmse:.4e}')
            events_avg_rmse = np.mean(events_rmse)
            logger.log(f'Average RMSE for all events: {events_avg_rmse:.4e}')
            if events_avg_rmse < best_rmse:
                best_rmse = events_avg_rmse
                best_hyperparameters['global_mass_loss_percent'] = global_mass_loss_percent
                logger.log(f'New best RMSE: {best_rmse:.4e} with global mass loss percent: {global_mass_loss_percent}')

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
