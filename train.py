import numpy as np
import os
import traceback
import torch

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset
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
        root_dir = train_dataset_parameters['root_dir']
        dataset_summary_file = train_dataset_parameters['dataset_summary_file']

        storage_mode = dataset_parameters['storage_mode']
        dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        dataset = dataset_class(
            root_dir=root_dir,
            dataset_summary_file=dataset_summary_file,
            nodes_shp_file=dataset_parameters['nodes_shp_file'],
            edges_shp_file=dataset_parameters['edges_shp_file'],
            spin_up_timesteps=dataset_parameters['spin_up_timesteps'],
            timesteps_from_peak=dataset_parameters['timesteps_from_peak'],
            inflow_boundary_edges=dataset_parameters['inflow_boundary_edges'],
            outflow_boundary_nodes=dataset_parameters['outflow_boundary_nodes'],
            # force_reload=True,
        )
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
        logger.log(f"Using loss function: {loss_func_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        num_epochs = train_config['num_epochs']

        # Training
        training_stats = TrainingStats(logger=logger)
        training_stats.start_train()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch in dataloader:
                optimizer.zero_grad()

                batch = batch.to(args.device)
                pred = model(batch)

                label = batch.y
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            training_stats.add_loss(epoch_loss)
            logger.log(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4e}')

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
            logger.log(f'Saved training stats to: {saved_metrics_path}')

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
