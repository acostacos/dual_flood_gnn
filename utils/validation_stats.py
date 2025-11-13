import numpy as np
import os
import time

from datetime import datetime
from numpy import ndarray
from torch import Tensor
from loss import GlobalMassConservationLoss, LocalMassConservationLoss
from data.dataset_normalizer import DatasetNormalizer
from typing import Optional, List
from utils import physics_utils

from . import Logger
from .metric_utils import RMSE, MAE, NSE, CSI

class ValidationStats:
    def __init__(self,
                 logger: Optional[Logger] = None,
                 normalizer: Optional[DatasetNormalizer] = None,
                 is_normalized: Optional[bool] = None,
                 delta_t: Optional[int] = None):
        self.normalizer = normalizer
        self.is_normalized = is_normalized
        self.delta_t = delta_t
        self.val_start_time = None
        self.val_end_time = None
        self.timestamps = []

        # ======== Water depth stats ========
        self.pred_list = []
        self.target_list = []

        # Overall stats
        self.rmse_list = []
        self.mae_list = []
        self.nse_list = []
        self.csi_list = []

        # Flooded cell stats
        self.rmse_flooded_list = []
        self.mae_flooded_list = []
        self.nse_flooded_list = []

        # ======== Water flow stats ========
        self.edge_pred_list = []
        self.edge_target_list = []

        # Overall stats
        self.edge_rmse_list = []
        self.edge_mae_list = []
        self.edge_nse_list = []

        # Flooded cell stats
        self.edge_rmse_flooded_list = []
        self.edge_mae_flooded_list = []
        self.edge_nse_flooded_list = []

        # ======== Physics-informed stats ========
        self.global_mass_loss_list = []
        self.local_mass_loss_list = []

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
    
    def start_validate(self):
        self.val_start_time = time.time()

    def end_validate(self):
        self.val_end_time = time.time()
    
    def get_inference_time(self):
        num_timesteps = max(len(self.pred_list), len(self.edge_pred_list))
        return (self.val_end_time - self.val_start_time) / num_timesteps
    
    def get_avg_rmse(self) -> float:
        return float(np.mean(self.rmse_list))

    def get_avg_mae(self) -> float:
        return float(np.mean(self.mae_list))

    def get_avg_nse(self) -> float:
        return float(np.mean(self.nse_list))

    def get_avg_edge_rmse(self) -> float:
        return float(np.mean(self.edge_rmse_list))

    def get_avg_edge_mae(self) -> float:
        return float(np.mean(self.edge_mae_list))

    def get_avg_edge_nse(self) -> float:
        return float(np.mean(self.edge_nse_list))

    def get_total_global_mass_loss(self) -> float:
        return float(np.sum(self.global_mass_loss_list))

    def get_total_local_mass_loss(self) -> float:
        return float(np.sum(self.local_mass_loss_list))

    def update_stats_for_timestep(self,
                                  pred: Tensor,
                                  target: Tensor,
                                  water_threshold: ndarray,
                                  timestamp: datetime = None):
        self.pred_list.append(pred)
        self.target_list.append(target)

        self.rmse_list.append(RMSE(pred, target))
        self.mae_list.append(MAE(pred, target))
        self.nse_list.append(NSE(pred, target))

        binary_pred = self.convert_water_depth_to_binary(pred, water_threshold=water_threshold)
        binary_target = self.convert_water_depth_to_binary(target, water_threshold=water_threshold)

        self.csi_list.append(CSI(binary_pred, binary_target))

        # Compute metrics for flooded areas only
        flooded_mask = binary_pred | binary_target
        flooded_pred, flooded_target = self.filter_by_water_threshold(pred, target, flooded_mask)

        self.rmse_flooded_list.append(RMSE(flooded_pred, flooded_target))
        self.mae_flooded_list.append(MAE(flooded_pred, flooded_target))
        self.nse_flooded_list.append(NSE(flooded_pred, flooded_target))

        if timestamp is not None:
            self.timestamps.append(timestamp)

    def convert_water_depth_to_binary(self, water_depth: Tensor, water_threshold: ndarray) -> Tensor:
        return (water_depth > water_threshold)

    def filter_by_water_threshold(self, pred: Tensor, target: Tensor, flooded_mask: Tensor):
        flooded_pred = pred[flooded_mask]
        flooded_target = target[flooded_mask]
        return flooded_pred, flooded_target

    def update_edge_stats_for_timestep(self, edge_pred: Tensor, edge_target: Tensor):
        self.edge_pred_list.append(edge_pred)
        self.edge_target_list.append(edge_target)

        self.edge_rmse_list.append(RMSE(edge_pred, edge_target))
        self.edge_mae_list.append(MAE(edge_pred, edge_target))
        self.edge_nse_list.append(NSE(edge_pred, edge_target))

    def update_physics_informed_stats_for_timestep(self,
                                                   pred: Tensor,
                                                   prev_node_pred: Tensor,
                                                   prev_edge_pred: Tensor,
                                                   databatch,
                                                   local_mass_nodes: List[int] = None):
        assert self.normalizer is not None and self.is_normalized is not None and self.delta_t is not None, \
            "normalizer, is_normalized, and delta_t must be set before updating physics-informed stats."

        global_mass_loss_func = GlobalMassConservationLoss(mode='test',
                                                           normalizer=self.normalizer,
                                                           is_normalized=self.is_normalized,
                                                           delta_t=self.delta_t)
        total_rainfall = physics_utils.get_total_rainfall(databatch)
        global_mass_loss = global_mass_loss_func(pred, prev_node_pred, prev_edge_pred, total_rainfall, databatch)
        self.global_mass_loss_list.append(global_mass_loss.cpu().item())

        local_mass_loss_func = LocalMassConservationLoss(mode='test',
                                                         normalizer=self.normalizer,
                                                         is_normalized=self.is_normalized,
                                                         delta_t=self.delta_t)
        rainfall = physics_utils.get_rainfall(databatch)
        local_nodes_mask = None
        if local_mass_nodes is not None:
            # Only compute local mass loss for specific nodes
            assert databatch.num_graphs == 1, 'For testing, assume there is only one graph per batch.'
            local_nodes_mask = np.isin(np.arange(databatch.num_nodes), local_mass_nodes)

        local_mass_loss = local_mass_loss_func(pred, prev_node_pred, prev_edge_pred, rainfall, databatch, local_nodes_mask)
        self.local_mass_loss_list.append(local_mass_loss.cpu().item())

    def print_stats_summary(self):
        if len(self.rmse_list) > 0:
            self.log(f'Average RMSE: {self.get_avg_rmse():.4e}')
        if len(self.rmse_flooded_list) > 0:
            self.log(f'Average RMSE (flooded): {np.mean(self.rmse_flooded_list):.4e}')
        if len(self.mae_list) > 0:
            self.log(f'Average MAE: {self.get_avg_mae():.4e}')
        if len(self.mae_flooded_list) > 0:
            self.log(f'Average MAE (flooded): {np.mean(self.mae_flooded_list):.4e}')
        if len(self.nse_list) > 0:
            self.log(f'Average NSE: {self.get_avg_nse():.4e}')
        if len(self.nse_flooded_list) > 0:
            self.log(f'Average NSE (flooded): {np.mean(self.nse_flooded_list):.4e}')
        if len(self.csi_list) > 0:
            self.log(f'Average CSI: {np.mean(self.csi_list):.4e}')

        if len(self.edge_rmse_list) > 0:
            self.log(f'\nAverage Edge RMSE: {self.get_avg_edge_rmse():.4e}')
        if len(self.edge_rmse_flooded_list) > 0:
            self.log(f'Average Edge RMSE (flooded): {np.mean(self.edge_rmse_flooded_list):.4e}')
        if len(self.edge_mae_list) > 0:
            self.log(f'Average Edge MAE: {self.get_avg_edge_mae():.4e}')
        if len(self.edge_mae_flooded_list) > 0:
            self.log(f'Average Edge MAE (flooded): {np.mean(self.edge_mae_flooded_list):.4e}')
        if len(self.edge_nse_list) > 0:
            self.log(f'Average Edge NSE: {self.get_avg_edge_nse():.4e}')
        if len(self.edge_nse_flooded_list) > 0:
            self.log(f'Average Edge NSE (flooded): {np.mean(self.edge_nse_flooded_list):.4e}')

        if len(self.global_mass_loss_list) > 0:
            self.log(f'\nTotal Global Mass Conservation Loss: {self.get_total_global_mass_loss():.4e}')
        if len(self.local_mass_loss_list) > 0:
            self.log(f'\nTotal Local Mass Conservation Loss: {self.get_total_local_mass_loss():.4e}')

        if self.val_start_time is not None and self.val_end_time is not None:
            self.log(f'\nValidation time: {self.val_end_time - self.val_start_time:.2f} seconds')
            self.log(f'Inference time for one timestep: {self.get_inference_time():.4f} seconds')

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        stats = {
            'timestamps': self.timestamps,
            'pred': np.array(self.pred_list),
            'target': np.array(self.target_list),
            'edge_pred': np.array(self.edge_pred_list),
            'edge_target': np.array(self.edge_target_list),
            'rmse': np.array(self.rmse_list),
            'rmse_flooded': np.array(self.rmse_flooded_list),
            'mae': np.array(self.mae_list),
            'mae_flooded': np.array(self.mae_flooded_list),
            'nse': np.array(self.nse_list),
            'nse_flooded': np.array(self.nse_flooded_list),
            'csi': np.array(self.csi_list),
            'edge_rmse': np.array(self.edge_rmse_list),
            'edge_rmse_flooded': np.array(self.edge_rmse_flooded_list),
            'edge_mae': np.array(self.edge_mae_list),
            'edge_mae_flooded': np.array(self.edge_mae_flooded_list),
            'edge_nse': np.array(self.edge_nse_list),
            'edge_nse_flooded': np.array(self.edge_nse_flooded_list),
            'global_mass_loss': np.array(self.global_mass_loss_list),
            'local_mass_loss': np.array(self.local_mass_loss_list),
            'inference_time': self.get_inference_time(),
        }
        np.savez(filepath, **stats)

        self.log(f'Saved metrics to: {filepath}')
