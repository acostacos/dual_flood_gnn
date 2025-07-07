import os
import numpy as np
import traceback
import torch

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset
from models import model_factory
from test import get_test_dataset_config, test_autoregressive_node_only
from train import train_w_global
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple
from utils import TrainingStats, ValidationStats, Logger, file_utils

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
    assert use_global_mass_loss, "Global mass loss must be enabled for training"
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

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    logger = Logger()

    try:
        logger.log('================================================')

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        HYPERPARAMETERS = {
            'global_mass_loss_percent': [0.003, 0.001, 0.0007, 0.0005, 0.0003, 0.0001, 0.00001],
        }

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
        best_hyperparameters = {
            'global_mass_loss_percent': None,
        }
        for global_mass_loss_percent in HYPERPARAMETERS['global_mass_loss_percent']:
            logger.log(f'\nRunning training with global mass loss percent: {global_mass_loss_percent}')

            # ============ Training Phase ============
            model = model_factory(args.model, **model_config)
            train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'])
            criterion = MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
            training_stats = TrainingStats(logger=None)

            training_stats.log = lambda x : None  # Suppress training stats logging to console
            train_w_global(model, train_dataloader, optimizer, criterion, training_stats, num_epochs, delta_t, global_mass_loss_percent, args.device)
            training_stats.log = logger.log  # Restore logging to console

            training_stats.print_stats_summary()

            # Save training stats and model
            curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if stats_dir is not None:
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)

                saved_metrics_path = os.path.join(stats_dir, f'{args.model}_{curr_date_str}_train_stats.npz')
                training_stats.save_stats(saved_metrics_path)

            if model_dir is not None:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_path = os.path.join(model_dir, f'{args.model}_{curr_date_str}.pt')
                torch.save(model.state_dict(), model_path)
                logger.log(f'Saved model to: {model_path}')

            # ============ Testing Phase ============
            logger.log('\nValidating model on test dataset...')

            events_rmse = []
            for event_idx, run_id in enumerate(test_dataset.hec_ras_run_ids):
                validation_stats = ValidationStats(logger=logger)
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
