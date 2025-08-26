import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from data import AutoregressiveFloodDataset, FloodEventDataset
from testing import DualAutoregressiveTester
from typing import Tuple
from utils import EarlyStopping

from .dual_regression_trainer import DualRegressionTrainer

class DualAutoRegressiveTrainer(DualRegressionTrainer):
    def __init__(self,
                 train_dataset: AutoregressiveFloodDataset,
                 val_dataset: FloodEventDataset,
                 init_num_timesteps: int = 1,
                 total_num_timesteps: int = 1,
                 early_stopping_patience: int = 15,
                #  curriculum_epochs: int = 10,
                 *args, **kwargs):
        assert isinstance(train_dataset, AutoregressiveFloodDataset), 'train_dataset must be an instance of AutoregressiveFloodEventDataset.'
        assert val_dataset is not None, 'val_dataset is required for autoregressive training.'
        assert init_num_timesteps <= total_num_timesteps, 'Initial number of timesteps must be less than or equal to total number of timesteps.'

        super().__init__(dataset=train_dataset, *args, **kwargs)

        if init_num_timesteps > 1 and self.num_epochs_dyn_loss > 0:
            self.training_stats.log('WARNING: not starting with a timestep of 1 for autoregressive training, while adjusting loss scaling rations dynamically is enabled. This may lead to unexpected behavior.')

        self.val_dataset = val_dataset
        self.init_num_timesteps = init_num_timesteps
        self.total_num_timesteps = total_num_timesteps
        self.patience = early_stopping_patience

        # Get non-boundary nodes/edges and threshold for metric computation
        self.non_boundary_nodes_mask = ~train_dataset.boundary_condition.boundary_nodes_mask
        self.non_boundary_edges_mask = ~train_dataset.boundary_condition.boundary_edges_mask

        # Get sliding window indices
        sliding_window_length = train_dataset.previous_timesteps + 1
        target_nodes_idx = train_dataset.DYNAMIC_NODE_FEATURES.index(train_dataset.NODE_TARGET_FEATURE)
        self.start_node_target_idx = train_dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

        target_edges_idx = train_dataset.DYNAMIC_EDGE_FEATURES.index(train_dataset.EDGE_TARGET_FEATURE)
        self.start_edge_target_idx = train_dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps

        early_stopping = EarlyStopping(patience=self.patience)
        for epoch in range(self.total_num_epochs):
            train_start_time = time.time()

            epoch_loss, pred_epoch_loss, edge_pred_epoch_loss = self._train_model(epoch, current_num_timesteps)

            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if epoch < self.num_epochs_dyn_loss:
                self.edge_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')
            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            non_dyn_epoch_num = epoch - self.num_epochs_dyn_loss
            if non_dyn_epoch_num < 0:
                continue

            val_node_rmse, val_edge_rmse = self._validate_model()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.log(f'\tValidation Edge RMSE: {val_edge_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
            self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

            # Previous critera to increase autoregression = non_dyn_epoch_num != 0 and non_dyn_epoch_num % self.curriculum_epochs == 0
            if early_stopping((val_node_rmse, val_edge_rmse), self.model):
                self.training_stats.log(f'\tEarly stopping triggered after {non_dyn_epoch_num + 1} epochs.')

                if current_num_timesteps < self.total_num_timesteps:
                    current_num_timesteps += 1
                    early_stopping = EarlyStopping(patience=self.patience)
                    self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps.')
                    continue

                self.training_stats.log('Training completed due to early stopping.')
                break

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float, float]:
        self.model.train()
        running_pred_loss = 0.0
        running_edge_pred_loss = 0.0
        if self.use_physics_loss:
            self._reset_epoch_physics_running_loss()

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)

            total_batch_loss = 0.0
            sliding_window = batch.x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            edge_sliding_window = batch.edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
            for i in range(current_num_timesteps):
                # Override graph data with sliding window
                # Only override non-boundary nodes to keep boundary conditions intact
                batch_non_boundary_nodes_mask = np.tile(self.non_boundary_nodes_mask, batch.num_graphs)
                batch.x[batch_non_boundary_nodes_mask, self.start_node_target_idx:self.end_node_target_idx] \
                    = sliding_window[batch_non_boundary_nodes_mask]

                # Only override non-boundary edges to keep boundary conditions intact
                batch_non_boundary_edges_mask = np.tile(self.non_boundary_edges_mask, batch.num_graphs)
                batch.edge_attr[batch_non_boundary_edges_mask, self.start_edge_target_idx:self.end_edge_target_idx] \
                    = edge_sliding_window[batch_non_boundary_edges_mask]

                pred, edge_pred = self.model(batch)
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch, i)

                label = batch.y[:, :, i]
                pred_loss = self.loss_func(pred, label)
                pred_loss = pred_loss * self.pred_loss_percent
                running_pred_loss += pred_loss.item()

                edge_label = batch.y_edge[:, :, i]
                edge_pred_loss = self.loss_func(edge_pred, edge_label)
                edge_pred_loss = self._scale_edge_pred_loss(epoch, pred_loss, edge_pred_loss)
                running_edge_pred_loss += edge_pred_loss.item()

                step_loss = pred_loss + edge_pred_loss

                if self.use_physics_loss:
                    prev_edge_pred = None if i == 0 else edge_sliding_window[:, [-1]]
                    physics_loss = self._get_epoch_physics_loss(epoch, pred, pred_loss, batch, prev_edge_pred)
                    step_loss = step_loss + physics_loss

                total_batch_loss = total_batch_loss + step_loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    next_sliding_window = torch.cat((sliding_window[:, 1:], pred), dim=1)
                    next_edge_sliding_window = torch.cat((edge_sliding_window[:, 1:], edge_pred), dim=1)

                    sliding_window = next_sliding_window
                    edge_sliding_window = next_edge_sliding_window

            total_batch_loss.backward()

            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.gradient_clip_value)

            self.optimizer.step()

        running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
        epoch_loss = running_loss / len(self.dataloader)
        pred_epoch_loss = running_pred_loss / len(self.dataloader)
        edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

        return epoch_loss, pred_epoch_loss, edge_pred_epoch_loss

    def _validate_model(self):
        val_tester = DualAutoregressiveTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        node_rmse = val_tester.get_avg_node_rmse()
        edge_rmse = val_tester.get_avg_edge_rmse()
        return node_rmse, edge_rmse

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask, :, timestep]

        # Only override inflow edges as outflow edges are predicted by the model
        batch_inflow_edges_mask = np.tile(self.inflow_edges_mask, batch.num_graphs)
        edge_pred[batch_inflow_edges_mask] = batch.y_edge[batch_inflow_edges_mask, :, timestep]
        return pred, edge_pred
