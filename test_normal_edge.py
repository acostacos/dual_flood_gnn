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
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
            'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
            'with_global_mass_loss': True,
            'with_local_mass_loss': True,
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
        normalizer = dataset.normalizer
        boundary_condition = dataset.boundary_condition
        is_normalized = dataset.is_normalized
        delta_t = dataset.timestep_interval

        # Get non-boundary nodes and threshold for metric computation
        non_boundary_nodes_mask = dataset.boundary_condition.get_non_boundary_nodes_mask()

        # Assume using the same area for all events in the dataset
        area_nodes_idx = dataset.STATIC_NODE_FEATURES.index('area')
        area = dataset[0].x.clone()[:, area_nodes_idx]
        if dataset.is_normalized:
            area = dataset.normalizer.denormalize('area', area)
        area = area[non_boundary_nodes_mask, None]
        threshold_per_cell = area * 0.05 # 5% of cell area

        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        log_test_config = {'rollout_start': rollout_start, 'rollout_timesteps': rollout_timesteps}
        logger.log(f'Using testing configuration: {log_test_config}')

        for i, run_id in enumerate(dataset.hec_ras_run_ids):
            logger.log(f'Validating on run {i + 1}/{len(dataset.hec_ras_run_ids)} with Run ID {run_id}')
            validation_stats = ValidationStats(logger=logger)
            validation_stats.start_validate()

            model.eval()
            with torch.no_grad():
                event_start_idx = dataset.event_start_idx[i] + rollout_start
                event_end_idx = dataset.event_start_idx[i + 1] if i + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
                if rollout_timesteps is not None:
                    event_end_idx = event_start_idx + rollout_timesteps
                    assert event_end_idx <= (dataset.event_start_idx[i + 1] if i + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps), \
                        f'Event end index {event_end_idx} exceeds dataset length {dataset.total_rollout_timesteps} for run ID {run_id}.'
                event_dataset = dataset[event_start_idx:event_end_idx]
                dataloader = DataLoader(event_dataset, batch_size=1) # Enforce batch size = 1 for autoregressive testing

                for graph in dataloader:
                    graph = graph.to(args.device)

                    pred, edge_pred = model(graph)

                    # Requires normalized physics-informed loss
                    validation_stats.update_physics_informed_stats_for_timestep(pred, graph,
                                                                                normalizer, boundary_condition,
                                                                                is_normalized=is_normalized, delta_t=delta_t)

                    label = graph.y
                    if dataset.is_normalized:
                        pred = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, pred)
                        label = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, label)

                    # Ensure water volume is non-negative
                    pred = torch.clip(pred, min=0)
                    label = torch.clip(label, min=0)

                    # Filter boundary conditions for metric computation
                    pred = pred[non_boundary_nodes_mask]
                    label = label[non_boundary_nodes_mask]

                    validation_stats.update_stats_for_timestep(pred.cpu(),
                                                    label.cpu(),
                                                    water_threshold=threshold_per_cell)

                    label_edge = graph.y_edge
                    if dataset.is_normalized:
                        edge_pred = dataset.normalizer.denormalize(dataset.EDGE_TARGET_FEATURE, edge_pred)
                        label_edge = dataset.normalizer.denormalize(dataset.EDGE_TARGET_FEATURE, label_edge)

                    validation_stats.update_edge_stats_for_timestep(edge_pred.cpu(), label_edge.cpu())

            validation_stats.end_validate()
            validation_stats.print_stats_summary()

            # Save validation stats
            output_dir = test_config['output_dir']
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Get filename from model path
                model_filename = os.path.splitext(os.path.basename(args.model_path))[0]  # Remove file extension
                saved_metrics_path = os.path.join(output_dir, f'{model_filename}_runid_{run_id}_test_metrics.npz') if output_dir is not None else None
                validation_stats.save_stats(saved_metrics_path)

        logger.log('================================================')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()