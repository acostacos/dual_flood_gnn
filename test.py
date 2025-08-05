import numpy as np
import traceback
import torch
import os

from argparse import ArgumentParser, Namespace
from data import FloodEventDataset, InMemoryFloodEventDataset
from models import model_factory
from torch_geometric.loader import DataLoader
from typing import Dict, Optional
from utils import Logger, file_utils
from utils.validation_stats import ValidationStats

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for validation')
    parser.add_argument('--model_path', required=True, default=None, help='Path to trained model file')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def get_test_dataset_config(base_datset_params: Dict, config: Dict) -> Dict:
    dataset_parameters = config['dataset_parameters']
    test_dataset_parameters = dataset_parameters['testing']
    test_dataset_config = {
        **base_datset_params,
        'mode': 'test',
        'dataset_summary_file': test_dataset_parameters['dataset_summary_file'],
        'event_stats_file': test_dataset_parameters['event_stats_file'],
        'with_global_mass_loss': True,
        'with_local_mass_loss': True,
    }
    return test_dataset_config

def test_autoregressive_node_only(model: torch.nn.Module,
                        dataset: FloodEventDataset,
                        event_idx: int,
                        validation_stats: ValidationStats,
                        rollout_start: int = 0,
                        rollout_timesteps: Optional[int] = None,
                        device: str = 'cpu',
                        include_physics_loss: bool = True):
    previous_timesteps = dataset.previous_timesteps

    # Get boundary condition masks
    boundary_nodes_mask = dataset.boundary_condition.boundary_nodes_mask
    non_boundary_nodes_mask = ~dataset.boundary_condition.boundary_nodes_mask

    # Assume using the same area for all events in the dataset
    area_nodes_idx = dataset.STATIC_NODE_FEATURES.index('area')
    area = dataset[0].x.clone()[:, area_nodes_idx]
    if dataset.is_normalized:
        area = dataset.normalizer.denormalize('area', area)
    area = area[non_boundary_nodes_mask, None]
    threshold_per_cell = area * 0.05 # 5% of cell area

    # Get sliding window indices
    target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
    sliding_window_length = previous_timesteps + 1
    start_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
    end_target_idx = start_target_idx + sliding_window_length

    validation_stats.start_validate()

    model.eval()
    with torch.no_grad():
        event_start_idx = dataset.event_start_idx[event_idx] + rollout_start
        event_end_idx = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
        if rollout_timesteps is not None:
            event_end_idx = event_start_idx + rollout_timesteps
            assert event_end_idx <= (dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps), \
                f'Event end index {event_end_idx} exceeds dataset length {dataset.total_rollout_timesteps} for event_idx {event_idx}.'
        event_dataset = dataset[event_start_idx:event_end_idx]
        dataloader = DataLoader(event_dataset, batch_size=1) # Enforce batch size = 1 for autoregressive testing

        sliding_window = dataset[event_start_idx].x.clone()[:, start_target_idx:end_target_idx]
        sliding_window = sliding_window.to(device)
        for graph in dataloader:
            graph = graph.to(device)

            # Override graph data with sliding window
            # Only override non-boundary nodes to keep boundary conditions intact
            graph.x[non_boundary_nodes_mask, start_target_idx:end_target_idx] = sliding_window[non_boundary_nodes_mask]

            pred = model(graph)

            # Override boundary conditions in predictions
            pred[boundary_nodes_mask] = graph.y[boundary_nodes_mask]

            sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)

            # Requires normalized physics-informed loss
            if include_physics_loss:
                # Requires normalized prediction for physics-informed loss
                prev_edge_pred = graph.global_mass_info['face_flow']
                validation_stats.update_physics_informed_stats_for_timestep(pred, graph, prev_edge_pred)

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

    validation_stats.end_validate()

def test_autoregressive(model: torch.nn.Module,
                        dataset: FloodEventDataset,
                        event_idx: int,
                        validation_stats: ValidationStats,
                        rollout_start: int = 0,
                        rollout_timesteps: Optional[int] = None,
                        device: str = 'cpu',
                        include_physics_loss: bool = True):
    previous_timesteps = dataset.previous_timesteps

    # Get non-boundary nodes/edges and threshold for metric computation
    boundary_nodes_mask = dataset.boundary_condition.boundary_nodes_mask
    non_boundary_nodes_mask = ~dataset.boundary_condition.boundary_nodes_mask
    non_boundary_edges_mask = ~dataset.boundary_condition.boundary_edges_mask
    inflow_edges_mask = dataset.boundary_condition.inflow_edges_mask

    # Assume using the same area for all events in the dataset
    area_nodes_idx = dataset.STATIC_NODE_FEATURES.index('area')
    area = dataset[0].x.clone()[:, area_nodes_idx]
    if dataset.is_normalized:
        area = dataset.normalizer.denormalize('area', area)
    area = area[non_boundary_nodes_mask, None]
    threshold_per_cell = area * 0.05 # 5% of cell area

    # Get sliding window indices
    sliding_window_length = previous_timesteps + 1
    target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
    start_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
    end_target_idx = start_target_idx + sliding_window_length

    target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
    start_target_edges_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
    end_target_edges_idx = start_target_edges_idx + sliding_window_length

    validation_stats.start_validate()

    model.eval()
    with torch.no_grad():
        event_start_idx = dataset.event_start_idx[event_idx] + rollout_start
        event_end_idx = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
        if rollout_timesteps is not None:
            event_end_idx = event_start_idx + rollout_timesteps
            dataset_event_length = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
            assert event_end_idx <= dataset_event_length, \
                f'Rollout length {event_end_idx} exceeds event length {dataset_event_length} for event {dataset.hec_ras_run_ids[event_idx]}.'
        event_dataset = dataset[event_start_idx:event_end_idx]
        dataloader = DataLoader(event_dataset, batch_size=1) # Enforce batch size = 1 for autoregressive testing

        sliding_window = dataset[event_start_idx].x.clone()[:, start_target_idx:end_target_idx]
        edge_sliding_window = dataset[event_start_idx].edge_attr.clone()[:, start_target_edges_idx:end_target_edges_idx]
        sliding_window, edge_sliding_window = sliding_window.to(device), edge_sliding_window.to(device)
        for i, graph in enumerate(dataloader):
            graph = graph.to(device)

            # Override graph data with sliding window
            # Only override non-boundary nodes to keep boundary conditions intact
            graph.x[non_boundary_nodes_mask, start_target_idx:end_target_idx] = sliding_window[non_boundary_nodes_mask]
            # Only override non-boundary edges to keep boundary conditions intact
            graph.edge_attr[non_boundary_edges_mask, start_target_edges_idx:end_target_edges_idx] = edge_sliding_window[non_boundary_edges_mask]

            pred, edge_pred = model(graph)

            # Override boundary conditions in predictions
            pred[boundary_nodes_mask] = graph.y[boundary_nodes_mask]
            # Only override inflow edges as outflow edges are predicted by the model
            edge_pred[inflow_edges_mask] = graph.y_edge[inflow_edges_mask]

            sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)
            edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred), dim=1)

            if include_physics_loss:
                # Requires normalized prediction for physics-informed loss
                prev_edge_pred = graph.global_mass_info['face_flow'] if i == 0 else edge_sliding_window[:, [-2]]
                validation_stats.update_physics_informed_stats_for_timestep(pred, graph, prev_edge_pred)

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

def run_test(model: torch.nn.Module,
             model_path: str,
             dataset: FloodEventDataset,
             logger: Logger,
             rollout_start: int = 0,
             rollout_timesteps: Optional[int] = None,
             output_dir: Optional[str] = None,
             device: str = 'cpu'):
    log_test_config = {'rollout_start': rollout_start, 'rollout_timesteps': rollout_timesteps}
    logger.log(f'Using testing configuration: {log_test_config}')

    is_dual_model = 'NodeEdgeGNN' in model.__class__.__name__

    avg_rmses = []
    avg_global_mass_losses = []
    avg_local_mass_losses = []
    if is_dual_model:
        avg_edge_rmses = []
    for event_idx, run_id in enumerate(dataset.hec_ras_run_ids):
        logger.log(f'Validating on run {event_idx + 1}/{len(dataset.hec_ras_run_ids)} with Run ID {run_id}')

        validation_stats = ValidationStats(logger=logger,
                                            previous_timesteps=dataset.previous_timesteps,
                                            normalizer=dataset.normalizer,
                                            is_normalized=dataset.is_normalized,
                                            delta_t=dataset.timestep_interval)

        if is_dual_model:
            test_autoregressive(model, dataset, event_idx, validation_stats, rollout_start, rollout_timesteps, device)
        else:
            test_autoregressive_node_only(model, dataset, event_idx, validation_stats, rollout_start, rollout_timesteps, device)
        validation_stats.print_stats_summary()

        avg_rmses.append(validation_stats.get_avg_rmse())
        avg_global_mass_losses.append(validation_stats.get_avg_global_mass_loss())
        avg_local_mass_losses.append(validation_stats.get_avg_local_mass_loss())
        if is_dual_model:
            avg_edge_rmses.append(validation_stats.get_avg_edge_rmse())

        # Save validation stats
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Get filename from model path
            model_filename = os.path.splitext(os.path.basename(model_path))[0]  # Remove file extension
            saved_metrics_path = os.path.join(output_dir, f'{model_filename}_runid_{run_id}_test_metrics.npz') if output_dir is not None else None
            validation_stats.save_stats(saved_metrics_path)

    logger.log(f'Average RMSE across events: {np.mean(avg_rmses):.4e}')
    if is_dual_model:
        logger.log(f'Average Edge RMSE across events: {np.mean(avg_edge_rmses):.4e}')
    logger.log(f'Average Global Mass Conservation Loss across events: {np.mean(avg_global_mass_losses):.4e}')
    logger.log(f'Average Local Mass Conservation Loss across events: {np.mean(avg_local_mass_losses):.4e}')

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
        dataset_config = {
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
        }
        dataset_config = get_test_dataset_config(dataset_config, config)
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
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model,
                 model_path=args.model_path,
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
