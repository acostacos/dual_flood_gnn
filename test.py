import numpy as np
import traceback
import torch
import os

from argparse import ArgumentParser, Namespace
from data import FloodEventDataset, InMemoryFloodEventDataset
from torch_geometric.loader import DataLoader
from train import model_factory
from utils import ValidationStats, Logger, file_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for validation')
    parser.add_argument('--model_path', required=True, default=None, help='Path to trained model file')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    test_config = config['testing_parameters']
    log_path = test_config['log_path']
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
        test_dataset_parameters = dataset_parameters['testing']
        dataset_summary_file = test_dataset_parameters['dataset_summary_file']
        event_stats_file = test_dataset_parameters['event_stats_file']
        dataset_config = {
            'mode': 'test',
            'root_dir': dataset_parameters['root_dir'],
            'dataset_summary_file': dataset_summary_file,
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'event_stats_file': event_stats_file,
            'features_stats_file': dataset_parameters['features_stats_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
            'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
            'inflow_boundary_edges': dataset_parameters['inflow_boundary_edges'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
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
        dataloader = DataLoader(dataset, batch_size=1) # Enforce batch size for autoregressive testing

        # Load model
        model_params = config['model_parameters'][args.model]
        previous_timesteps = dataset.previous_timesteps
        base_model_params = {
            'static_node_features': dataset.num_static_node_features,
            'dynamic_node_features': dataset.num_dynamic_node_features,
            'static_edge_features': dataset.num_static_edge_features,
            'dynamic_edge_features': dataset.num_dynamic_edge_features,
            'previous_timesteps': previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        logger.log(f'Using model checkpoint for {args.model}: {args.model_path}')
        logger.log(f'Using model configuration: {model_config}')

        # Testing
        validation_stats = ValidationStats(logger=logger)
        validation_stats.start_validate()

        # Get non-boundary nodes for filtering
        non_boundary_nodes = torch.ones(dataset[0].x.shape[0], dtype=torch.bool)
        non_boundary_nodes[dataset[0].boundary_nodes] = False

        # Get cell area for theshold calculation
        area_nodes_idx = dataset.STATIC_NODE_FEATURES.index('area')
        area = dataset[0].x.clone()[:, area_nodes_idx]
        denorm_area = dataset._denormalize_features('area', area)
        denorm_area = denorm_area[non_boundary_nodes, None]
        threshold_per_cell = denorm_area * 0.05 # 5% of cell area

        model.eval()
        with torch.no_grad():
            target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
            sliding_window_length = previous_timesteps + 1
            start_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
            end_target_idx = start_target_idx + sliding_window_length
            sliding_window = dataset[0].x.clone()[:, start_target_idx:end_target_idx]
            sliding_window = sliding_window.to(args.device)

            for graph in dataloader:
                graph = graph.to(args.device)

                # Override graph data with sliding window
                graph.x[:, start_target_idx:end_target_idx] = sliding_window

                pred = model(graph)
                sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)

                label = graph.y
                if dataset.normalize:
                    pred = dataset._denormalize_features(dataset.NODE_TARGET_FEATURE, pred)
                    label = dataset._denormalize_features(dataset.NODE_TARGET_FEATURE, label)

                # Ensure water volume is non-negative
                pred = torch.clip(pred, min=0)
                label = torch.clip(label, min=0)

                # Filter boundary conditions for metric computation
                pred = pred[non_boundary_nodes]
                label = label[non_boundary_nodes]

                validation_stats.update_stats_for_epoch(pred.cpu(),
                                                  label.cpu(),
                                                  water_threshold=threshold_per_cell)

        validation_stats.end_validate()
        validation_stats.print_stats_summary()

        # Save validation stats
        output_dir = test_config['output_dir']
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            saved_metrics_path = os.path.join(output_dir, f'{args.model_path}_test_metrics.npz') if args.output_dir is not None else None
            validation_stats.save_stats(saved_metrics_path)

        logger.log('================================================')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
