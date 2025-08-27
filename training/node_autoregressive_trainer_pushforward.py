import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from data import AutoregressiveFloodDataset
from testing import NodeAutoregressiveTester
from typing import Tuple
from utils import EarlyStopping

from .base_autoregressive_trainer import BaseAutoregressiveTrainer
from .physics_informed_trainer import PhysicsInformedTrainer

class NodeAutoregressiveTrainer(BaseAutoregressiveTrainer, PhysicsInformedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: AutoregressiveFloodDataset = self.dataloader.dataset
        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask
        self.non_boundary_nodes_mask = ~ds.boundary_condition.boundary_nodes_mask

        # Get sliding window indices
        sliding_window_length = ds.previous_timesteps + 1
        target_nodes_idx = ds.DYNAMIC_NODE_FEATURES.index(ds.NODE_TARGET_FEATURE)
        self.start_node_target_idx = ds.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

    def train(self):
        self.training_stats.start_train()

        for epoch in range(self.total_num_epochs):
            train_start_time = time.time()

            epoch_loss, pred_epoch_loss, stab_epoch_loss, epoch_grad_norm = self._train_model(epoch)

            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tStability Loss: {stab_epoch_loss:.4e}'
            logging_str += f'\n\tGrad Norm: {epoch_grad_norm:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('stability_loss', stab_epoch_loss)

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            non_dyn_epoch_num = epoch - self.num_epochs_dyn_loss
            if self.use_physics_loss and non_dyn_epoch_num < 0:
                continue

            val_node_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)

            # Previous critera to increase autoregression = non_dyn_epoch_num != 0 and non_dyn_epoch_num % self.curriculum_epochs == 0
            if self.early_stopping(val_node_rmse, self.model):
                self.training_stats.log(f'\tEarly stopping triggered after {non_dyn_epoch_num + 1} epochs.')
                self.training_stats.log('Training completed due to early stopping.')
                break

        self.training_stats.end_train()
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_pred_loss = 0.0
        running_stab_loss = 0.0
        running_grad_norm = 0.0
        if self.use_physics_loss:
            self._reset_epoch_physics_running_loss()

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            x, edge_index = batch.x[:, :, 0], batch.edge_index

            # Prediction for timestep t
            edge_attr = batch.edge_attr[:, :, 0]
            pred = self.model(x, edge_index, edge_attr)
            pred = self._override_pred_bc(pred, batch, 0)

            pred_loss = self._compute_node_loss(pred, batch, 0)
            # pred_loss = pred_loss * self.pred_loss_percent
            running_pred_loss += pred_loss.item()

            one_step_loss = pred_loss

            # Prediction for timestep t+1
            sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            next_sliding_window = torch.cat((sliding_window[:, 1:], pred), dim=1)
            next_x = torch.concat([x[:, :self.start_node_target_idx], next_sliding_window, x[:, self.end_node_target_idx:]], dim=1)

            next_edge_attr = batch.edge_attr[:, :, 1]
            next_pred = self.model(next_x, edge_index, next_edge_attr)
            next_pred = self._override_pred_bc(next_pred, batch, 1)

            next_pred_loss = self._compute_node_loss(next_pred, batch, 1)
            # next_pred_loss = next_pred_loss * self.pred_loss_percent
            running_stab_loss += next_pred_loss.item()

            stability_loss = next_pred_loss

            if self.use_physics_loss:
                physics_loss = self._get_epoch_physics_loss(epoch, pred, pred_loss, batch, None)
                step_loss = step_loss + physics_loss

            total_batch_loss = one_step_loss + stability_loss
            total_batch_loss.backward()
            running_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
            self._clip_gradients()
            self.optimizer.step()

        running_loss = self._get_epoch_total_running_loss(running_pred_loss)
        epoch_loss = running_loss / len(self.dataloader)
        pred_epoch_loss = running_pred_loss / len(self.dataloader)
        stab_epoch_loss = running_stab_loss / len(self.dataloader)
        epoch_grad_norm = running_grad_norm / len(self.dataloader)

        return epoch_loss, pred_epoch_loss, stab_epoch_loss, epoch_grad_norm

    def validate(self):
        val_tester = NodeAutoregressiveTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        node_rmse = val_tester.get_avg_node_rmse()
        return node_rmse

    def _compute_node_loss(self, pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y[:, :, timestep]
        return self.loss_func(pred, label)

    def _override_pred_bc(self, pred: Tensor, batch, timestep: int) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask, :, timestep]
        return pred
