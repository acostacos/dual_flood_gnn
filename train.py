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
            # force_reload=True,
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
        node_loss_weight = loss_func_parameters['node_loss_weight']
        loss_func_name = criterion.__name__ if hasattr(criterion, '__name__') else criterion.__class__.__name__
        logger.log(f"Using {loss_func_name} loss for nodes with weight {node_loss_weight}")
        if use_global_mass_loss:
            global_mass_loss_weight = loss_func_parameters['global_mass_loss_weight']
            logger.log(f'Using global mass conservation loss with weight {global_mass_loss_weight}')

        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        num_epochs = train_config['num_epochs']
        delta_t = dataset.timestep_interval

        # Training
        training_stats = TrainingStats(logger=logger)
        training_stats.start_train()
        for epoch in range(num_epochs):
            model.train()
            running_pred_loss = 0.0
            running_global_physics_loss = 0.0

            for batch in dataloader:
                optimizer.zero_grad()

                batch = batch.to(args.device)
                pred = model(batch)

                label = batch.y

                if use_global_mass_loss:
                    pred_loss = node_loss_weight * criterion(pred, label)
                    global_physics_loss =  global_mass_loss_weight * global_mass_conservation_loss(pred,
                                                                                                   batch,
                                                                                                   delta_t=delta_t)
                    loss = pred_loss + global_physics_loss
                    running_pred_loss += pred_loss.item()
                    running_global_physics_loss += global_physics_loss.item()
                else:
                    loss = criterion(pred, label)
                    running_pred_loss += loss.item()

                loss.backward()
                optimizer.step()

            running_loss = running_pred_loss + running_global_physics_loss
            epoch_loss = running_loss / len(dataloader)
            pred_epoch_loss = running_pred_loss / len(dataloader)
            global_physics_epoch_loss = running_global_physics_loss / len(dataloader)

            logging_str = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4e}'
            if use_global_mass_loss:
                logging_str += f', Prediction Loss: {pred_epoch_loss:.4e}, Global Physics Loss: {global_physics_epoch_loss:.4e}'
            logger.log(logging_str)

            training_stats.add_loss(epoch_loss)
            training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            training_stats.add_loss_component('global_physics_loss', global_physics_epoch_loss)

        training_stats.end_train()
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
