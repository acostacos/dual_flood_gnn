import numpy as np
import os
import traceback
import torch
import gc
import random

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset, \
    AutoregressiveFloodEventDataset, InMemoryAutoregressiveFloodEventDataset
from models import model_factory
from test import get_test_dataset_config, run_test
from torch.nn import MSELoss
from training import NodeRegressionTrainer, DualRegressionTrainer, DualAutoRegressiveTrainer
from typing import Dict, Literal, Optional, Tuple
from utils import Logger, file_utils, train_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--with_test", type=bool, default=False, help='Whether to run test after training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def get_dataset_class(storage_mode: Literal['memory', 'disk'], autoregressive: bool = False) -> type:
    if autoregressive:
        if storage_mode == 'memory':
            return InMemoryAutoregressiveFloodEventDataset
        elif storage_mode == 'disk':
            return AutoregressiveFloodEventDataset

    if storage_mode == 'memory':
        return InMemoryFloodEventDataset
    elif storage_mode == 'disk':
        return FloodEventDataset

    raise ValueError(f'Dataset class is not defined.')

def load_dataset(config: Dict, args: Namespace, logger: Logger) -> Tuple[FloodEventDataset, Optional[FloodEventDataset]]:
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    train_dataset_parameters = dataset_parameters['training']
    loss_func_parameters = config['loss_func_parameters']
    base_datset_config = {
        'root_dir': root_dir,
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
        'with_global_mass_loss': loss_func_parameters['use_global_mass_loss'],
        'with_local_mass_loss': loss_func_parameters['use_local_mass_loss'],
        'debug': args.debug,
        'logger': logger,
        'force_reload': True,
    }

    dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    event_stats_file = train_dataset_parameters['event_stats_file']
    storage_mode = dataset_parameters['storage_mode']

    training_parameters = config['training_parameters']
    if 'NodeEdgeGNN' in args.model and training_parameters.get('autoregressive', False):
        # Split dataset into training and validation sets for autoregressive training
        percent_validation = config['training_parameters'].get('percent_validation', 0.1)
        logger.log(f'Splitting dataset into training and validation sets with {percent_validation * 100}% for validation')
        train_summary_file, val_summary_file = train_utils.split_dataset_events(root_dir, dataset_summary_file, percent_validation)

        train_dataset_config = {
            'mode': 'train',
            'dataset_summary_file': train_summary_file,
            'event_stats_file': f'train_split_{event_stats_file}',
            'num_label_timesteps': training_parameters['autoregressive_timesteps'],
            **base_datset_config,
        }
        logger.log(f'Using training dataset configuration: {train_dataset_config}')
        train_dataset_class = get_dataset_class(storage_mode, autoregressive=True)
        train_dataset = train_dataset_class(**train_dataset_config)

        val_dataset_config = {
            'mode': 'test',
            'dataset_summary_file': val_summary_file,
            'event_stats_file': f'val_split_{event_stats_file}',
            **base_datset_config,
        }
        logger.log(f'Using validation dataset configuration: {val_dataset_config}')
        test_dataset_class = get_dataset_class(storage_mode, autoregressive=False)
        val_dataset = test_dataset_class(**val_dataset_config)

        logger.log(f'Split dataset into {len(train_dataset)} training samples and {len(val_dataset)} validation samples')
        return train_dataset, val_dataset

    dataset_config = {
        'mode': 'train',
        'dataset_summary_file': dataset_summary_file,
        'event_stats_file': event_stats_file,
        **base_datset_config,
    }
    logger.log(f'Using dataset configuration: {dataset_config}')

    dataset_class = get_dataset_class(storage_mode)
    dataset = dataset_class(**dataset_config)
    logger.log(f'Loaded dataset with {len(dataset)} samples')
    return dataset, None

def run_train(model: torch.nn.Module,
              model_name: str,
              train_dataset: FloodEventDataset,
              logger: Logger,
              config: Dict,
              val_dataset: Optional[FloodEventDataset] = None,
              stats_dir: Optional[str] = None,
              model_dir: Optional[str] = None,
              device: str = 'cpu') -> str:
        train_config = config['training_parameters']
        loss_func_parameters = config['loss_func_parameters']

        # Loss function and optimizer
        num_epochs = train_config['num_epochs']
        num_epochs_dyn_loss = train_config['num_epochs_dyn_loss']
        batch_size = train_config['batch_size']
        gradient_clip_value = train_config['gradient_clip_value']
        log_train_config = {'num_epochs': num_epochs, 'num_epochs_dyn_loss': num_epochs_dyn_loss, 'batch_size': batch_size, 'learning_rate': train_config['learning_rate'], 'weight_decay': train_config['weight_decay'], 'gradient_clip_value': gradient_clip_value }
        logger.log(f'Using training configuration: {log_train_config}')
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        delta_t = train_dataset.timestep_interval

        criterion = MSELoss()
        loss_func_name = criterion.__name__ if hasattr(criterion, '__name__') else criterion.__class__.__name__
        logger.log(f"Using {loss_func_name} loss for nodes")

        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
        trainer_params = {
            'model': model,
            'optimizer': optimizer,
            'loss_func': criterion,
            'use_global_loss': use_global_mass_loss,
            'use_local_loss': use_local_mass_loss,
            'delta_t': delta_t,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_epochs_dyn_loss': num_epochs_dyn_loss,
            'gradient_clip_value': gradient_clip_value,
            'logger': logger,
            'device': device,
        }
        if use_global_mass_loss:
            global_mass_loss_scale = loss_func_parameters['global_mass_loss_scale']
            global_mass_loss_percent = loss_func_parameters['global_mass_loss_percent']
            logger.log(f'Using global mass conservation loss with initial scale {global_mass_loss_scale} and loss percentage {global_mass_loss_percent}')
            trainer_params.update({
                'global_mass_loss_scale': global_mass_loss_scale,
                'global_mass_loss_percent': global_mass_loss_percent,
            })
        if use_local_mass_loss:
            local_mass_loss_scale = loss_func_parameters['local_mass_loss_scale']
            local_mass_loss_percent = loss_func_parameters['local_mass_loss_percent']
            logger.log(f'Using local mass conservation loss with inital scale {local_mass_loss_scale} and loss percentage {local_mass_loss_percent}')
            trainer_params.update({
                'local_mass_loss_scale': local_mass_loss_scale,
                'local_mass_loss_percent': local_mass_loss_percent,
            })

        if 'NodeEdgeGNN' in model_name:
            edge_pred_loss_scale = loss_func_parameters['edge_pred_loss_scale']
            edge_pred_loss_percent = loss_func_parameters['edge_pred_loss_percent']
            logger.log(f'Using edge prediction loss with initial scale {edge_pred_loss_scale} and loss percentage {edge_pred_loss_percent}')
            trainer_params.update({
                'edge_pred_loss_scale': edge_pred_loss_scale,
                'edge_pred_loss_percent': edge_pred_loss_percent,
            })

            if train_config.get('autoregressive', False):
                assert val_dataset is not None, "Validation dataset is required for autoregressive training"
                num_timesteps = train_config['autoregressive_timesteps']
                curriculum_epochs = train_config['curriculum_epochs']
                logger.log(f'Using autoregressive training with intervals of {num_timesteps} timessteps and curriculum learning for {curriculum_epochs} epochs')
                trainer_params.update({
                    'num_timesteps': num_timesteps,
                    'curriculum_epochs': curriculum_epochs,
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                })

                trainer = DualAutoRegressiveTrainer(**trainer_params)
            else:
                trainer_params.update({ 'dataset': train_dataset })
                trainer = DualRegressionTrainer(**trainer_params)
        else:
            trainer_params.update({ 'dataset': train_dataset })
            trainer = NodeRegressionTrainer(**trainer_params)
        trainer.train()

        trainer.print_stats_summary()

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
            trainer.save_stats(saved_metrics_path)

        model_path = f'{model_name}_{curr_date_str}.pt'
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
            trainer.save_model(model_path)

        return model_path

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Dataset
        train_dataset, val_dataset = load_dataset(config, args, logger)

        # Model
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
        model = model_factory(args.model, **model_config)
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')

        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        model_path = run_train(model=model,
                               model_name=args.model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               logger=logger,
                               config=config,
                               stats_dir=stats_dir,
                               model_dir=model_dir,
                               device=args.device)

        logger.log('================================================')

        if not args.with_test:
            return

        # =================== Testing ===================
        logger.log(f'Starting testing for model: {model_path}')

        dataset_parameters = config['dataset_parameters']
        base_datset_config = {
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
            'debug': args.debug,
            'logger': logger,
            'force_reload': True,
        }
        test_dataset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using test dataset configuration: {test_dataset_config}')

        # Clear memory before loading test dataset
        del dataset
        gc.collect()

        storage_mode = dataset_parameters['storage_mode']
        dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        dataset = dataset_class(**test_dataset_config)
        logger.log(f'Loaded test dataset with {len(dataset)} samples')

        logger.log(f'Using model checkpoint for {args.model}: {model_path}')
        logger.log(f'Using model configuration: {model_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model,
                 model_path=model_path,
                 dataset=dataset,
                 logger=logger,
                 rollout_start=rollout_start,
                 rollout_timesteps=rollout_timesteps,
                 output_dir=output_dir,
                 device=args.device)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
