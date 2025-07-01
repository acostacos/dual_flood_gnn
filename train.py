import numpy as np
import os
import traceback
import torch

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset
from loss import global_mass_conservation_loss
from models import GAT, GCN
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from typing import Callable
from utils import TrainingStats, Logger, file_utils

torch.serialization.add_safe_globals([datetime])

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def model_factory(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

def train_base(model: torch.nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: Callable,
               training_stats: TrainingStats,
               num_epochs: int = 100,
               device: str = 'cpu'):
    training_stats.start_train()
    for epoch in range(num_epochs):
        model.train()
        running_pred_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            batch = batch.to(device)
            pred = model(batch)

            label = batch.y
            loss = criterion(pred, label)
            running_pred_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = running_pred_loss / len(dataloader)
        logging_str = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4e}'
        training_stats.log(logging_str)

        training_stats.add_loss(epoch_loss)
    training_stats.end_train()

def train_w_global(model: torch.nn.Module,
                   dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: Callable,
                   training_stats: TrainingStats,
                   num_epochs: int = 100,
                   delta_t: int = 30,
                   global_mass_loss_percent: float = 0.2,
                   global_mass_loss_weight: float = 0.01,
                   device: str = 'cpu'):
    DYNAMIC_LOSS_WEIGHT_NUM_EPOCHS = num_epochs * 0.2

    # pred_loss_percent = 1.0 - global_mass_loss_percent
    scaled_loss_ratios = []
    training_stats.start_train()
    for epoch in range(num_epochs):
        model.train()
        running_pred_loss = 0.0
        running_global_physics_loss = 0.0

        # Dynamic loss weight adjustment
        running_absolute_loss_ratio = 0.0
        running_scaled_loss_ratio = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            batch = batch.to(device)
            pred = model(batch)

            label = batch.y
            pred_loss = criterion(pred, label)
            # pred_loss =  pred_loss * pred_loss_percent
            running_pred_loss += pred_loss.item()

            global_physics_loss = global_mass_conservation_loss(pred, batch, delta_t=delta_t)
            absolute_loss_ratio = pred_loss / (global_physics_loss + 1e-8)
            running_absolute_loss_ratio += absolute_loss_ratio.item()

            scaled_global_physics_loss = global_mass_loss_weight * global_physics_loss
            scaled_loss_ratio = pred_loss / (scaled_global_physics_loss + 1e-8)
            running_scaled_loss_ratio  += scaled_loss_ratio.item()
            # scaled_global_physics_loss = scaled_global_physics_loss * global_mass_loss_percent
            running_global_physics_loss += scaled_global_physics_loss.item()

            loss = pred_loss + scaled_global_physics_loss

            loss.backward()
            optimizer.step()

        running_loss = running_pred_loss + running_global_physics_loss
        epoch_loss = running_loss / len(dataloader)
        pred_epoch_loss = running_pred_loss / len(dataloader)
        global_physics_epoch_loss = running_global_physics_loss / len(dataloader)
        epoch_scaled_loss_ratio = running_scaled_loss_ratio / len(dataloader)

        logging_str = f'Epoch [{epoch + 1}/{num_epochs}]\n'
        logging_str += f'\tLoss: {epoch_loss:.4e}\n'
        logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}\n'
        logging_str += f'\tGlobal Physics Loss: {global_physics_epoch_loss:.4e}\n'
        logging_str += f'\tAverage Loss Ratio: {epoch_scaled_loss_ratio:.4e}'
        training_stats.log(logging_str)

        training_stats.add_loss(epoch_loss)
        training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
        training_stats.add_loss_component('global_physics_loss', global_physics_epoch_loss)
        scaled_loss_ratios.append(epoch_scaled_loss_ratio)

        if epoch < DYNAMIC_LOSS_WEIGHT_NUM_EPOCHS:
            epoch_absolute_loss_ratio = running_absolute_loss_ratio / len(dataloader)
            global_mass_loss_weight = epoch_absolute_loss_ratio
            training_stats.log(f'\tAdjusted Global Mass Loss Weight to {global_mass_loss_weight:.4e}')

    training_stats.end_train()
    training_stats.add_additional_info('loss_ratios', scaled_loss_ratios)

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

        # Dataset
        dataset_parameters = config['dataset_parameters']
        train_dataset_parameters = dataset_parameters['training']
        dataset_summary_file = train_dataset_parameters['dataset_summary_file']
        event_stats_file = train_dataset_parameters['event_stats_file']
        loss_func_parameters = config['loss_func_parameters']
        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        dataset_config = {
            'mode': 'train',
            'root_dir': dataset_parameters['root_dir'],
            'dataset_summary_file': dataset_summary_file,
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'event_stats_file': event_stats_file,
            'features_stats_file': dataset_parameters['features_stats_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
            'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
            'with_global_mass_loss': use_global_mass_loss,
        }
        logger.log(f'Using dataset configuration: {dataset_config}')

        storage_mode = dataset_parameters['storage_mode']
        dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        dataset = dataset_class(
            **dataset_config,
            debug=args.debug,
            logger=logger,
            force_reload=True,
        )
        logger.log(f'Loaded dataset with {len(dataset)} samples')
        dataloader = DataLoader(dataset, batch_size=train_config['batch_size'])

        # Model
        model_params = config['model_parameters'][args.model]
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
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')

        # Loss function and optimizer
        criterion = MSELoss()
        loss_func_name = criterion.__name__ if hasattr(criterion, '__name__') else criterion.__class__.__name__
        logger.log(f"Using {loss_func_name} loss for nodes")
        if use_global_mass_loss:
            global_mass_loss_percent = loss_func_parameters['global_mass_loss_percent']
            init_global_loss_weight = loss_func_parameters['init_global_loss_weight']
            logger.log(f'Using global mass conservation loss with initial weight {init_global_loss_weight} and target percentage {global_mass_loss_percent}')

        log_train_config = {'num_epochs': train_config['num_epochs'], 'batch_size': train_config['batch_size'], 'learning_rate': train_config['learning_rate'], 'weight_decay': train_config['weight_decay'] }
        logger.log(f'Using training configuration: {log_train_config}')
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        num_epochs = train_config['num_epochs']
        training_stats = TrainingStats(logger=logger)

        if use_global_mass_loss:
            delta_t = dataset.timestep_interval
            train_w_global(model, dataloader, optimizer, criterion, training_stats, num_epochs, delta_t, global_mass_loss_percent, init_global_loss_weight, args.device)
        else:
            train_base(model, dataloader, optimizer, criterion, training_stats, num_epochs, args.device)

        training_stats.print_stats_summary()

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        stats_dir = train_config['stats_dir']
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{args.model}_{curr_date_str}_train_stats.npz')
            training_stats.save_stats(saved_metrics_path)

        model_dir = train_config['model_dir']
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{args.model}_{curr_date_str}.pt')
            torch.save(model.state_dict(), model_path)
            logger.log(f'Saved model to: {model_path}')

        logger.log('================================================')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
