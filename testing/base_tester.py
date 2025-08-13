import os
import numpy as np

from torch.nn import Module
from data import FloodEventDataset
from utils import Logger
from utils.validation_stats import ValidationStats
from typing import List, Optional

class BaseTester:
    def __init__(self,
                 model: Module,
                 dataset: FloodEventDataset,
                 rollout_start: int = 0,
                 rollout_timesteps: Optional[int] = None,
                 include_physics_loss: bool = True,
                 logger: Logger = None,
                 device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.rollout_start = rollout_start
        self.rollout_timesteps = rollout_timesteps
        self.include_physics_loss = include_physics_loss
        self.logger = logger
        self.device = device
        self.events_validation_stats: List[ValidationStats] = []

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_nodes_mask = dataset.boundary_condition.boundary_nodes_mask
        self.non_boundary_nodes_mask = ~dataset.boundary_condition.boundary_nodes_mask
        self.non_boundary_edges_mask = ~dataset.boundary_condition.boundary_edges_mask
        self.inflow_edges_mask = dataset.boundary_condition.inflow_edges_mask

        # Assume using the same area for all events in the dataset
        area_nodes_idx = dataset.STATIC_NODE_FEATURES.index('area')
        area = dataset[0].x.clone()[:, area_nodes_idx]
        if dataset.is_normalized:
            area = dataset.normalizer.denormalize('area', area)
        area = area[self.non_boundary_nodes_mask, None]
        self.threshold_per_cell = area * 0.05 # 5% of cell area

        # Get sliding window indices
        previous_timesteps = dataset.previous_timesteps
        sliding_window_length = previous_timesteps + 1

        target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
        self.start_node_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

        target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
        self.start_edge_target_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

    def test(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_avg_node_rmse(self) -> float:
        rmses = [stat.get_avg_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else 0.0

    def get_avg_global_mass_loss(self) -> float:
        losses = [stat.get_avg_global_mass_loss() for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def get_avg_local_mass_loss(self) -> float:
        losses = [stat.get_avg_local_mass_loss() for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def save_stats(self, output_dir: str, stats_filename_prefix: Optional[str] = None):
        for event_idx, run_id in enumerate(self.dataset.hec_ras_run_ids):
            validation_stats = self.events_validation_stats[event_idx]
            saved_metrics_path = os.path.join(output_dir, f'{stats_filename_prefix}_runid_{run_id}_test_metrics.npz')
            validation_stats.save_stats(saved_metrics_path)
