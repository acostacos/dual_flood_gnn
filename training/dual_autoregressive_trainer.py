import os
import time
import torch

from contextlib import redirect_stdout
from torch import Tensor
from testing import DualAutoregressiveTester
from typing import Tuple, Callable
from utils import EarlyStopping, LossScaler, train_utils

from .node_autoregressive_trainer import NodeAutoregressiveTrainer
from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer

class DualAutoregressiveTrainer(NodeAutoregressiveTrainer, EdgeAutoregressiveTrainer):
    def __init__(self,
                 edge_loss_func: Callable,
                 edge_pred_loss_scale: float = 1.0,
                 edge_loss_weight: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edge_loss_func = edge_loss_func
        self.edge_loss_weight = edge_loss_weight
        self.edge_loss_scaler = LossScaler(initial_scale=edge_pred_loss_scale)

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps
        current_timestep_epochs = 0

        for epoch in range(self.num_epochs):
            train_start_time = time.time()

            train_losses = self._train_model(epoch, current_num_timesteps)
            epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = train_losses

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if self.use_physics_loss:
                self._log_epoch_physics_loss(global_mass_epoch_loss, local_mass_epoch_loss)

            self._update_loss_scaler_for_epoch(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            val_node_rmse, val_edge_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.log(f'\tValidation Edge RMSE: {val_edge_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
            self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

            current_timestep_epochs += 1

            is_early_stopped = self.early_stopping((val_node_rmse, val_edge_rmse), self.model)
            is_max_exceeded = self.max_curriculum_epochs is not None and current_timestep_epochs >= self.max_curriculum_epochs
            if (is_early_stopped or is_max_exceeded) and current_num_timesteps < self.total_num_timesteps:
                self.training_stats.log(f'\tCurriculum learning for {current_num_timesteps} steps ended after {current_timestep_epochs} epochs.')
                current_num_timesteps += 1
                current_timestep_epochs = 0
                self.early_stopping = EarlyStopping(patience=self.early_stopping.patience)
                self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps.')
                self.lr_scheduler.step()
                self.training_stats.log(f'\tDecayed learning rate to {self.lr_scheduler.get_last_lr()[0]:.4e}.')
                continue

            if is_early_stopped:
                self.training_stats.log('Training completed due to early stopping.')
                break

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float, float, float, float]:
        self.model.train()

        running_pred_loss = 0.0
        running_edge_pred_loss = 0.0
        running_global_mass_loss = 0.0
        running_local_mass_loss = 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            x, edge_attr, edge_index = batch.x[:, :, 0], batch.edge_attr[:, :, 0], batch.edge_index

            total_batch_loss = 0.0
            total_batch_pred_loss = 0.0
            total_batch_edge_pred_loss = 0.0
            total_batch_global_mass_loss = 0.0
            total_batch_local_mass_loss = 0.0

            sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            edge_sliding_window = edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
            for i in range(current_num_timesteps):
                x, edge_attr = batch.x[:, :, i], batch.edge_attr[:, :, i]

                # Override graph data with sliding window
                x = torch.concat([x[:, :self.start_node_target_idx], sliding_window, x[:, self.end_node_target_idx:]], dim=1)
                edge_attr = torch.concat([edge_attr[:, :self.start_edge_target_idx], edge_sliding_window, edge_attr[:, self.end_edge_target_idx:]], dim=1)

                pred_diff, edge_pred_diff = self.model(x, edge_index, edge_attr)
                pred_diff, edge_pred_diff = self._override_pred_bc(pred_diff, edge_pred_diff, batch, i)

                pred_loss = self._compute_node_loss(pred_diff, batch, i)
                pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                total_batch_pred_loss += pred_loss.item()

                edge_pred_loss = self._compute_edge_loss(edge_pred_diff, batch, i)
                edge_pred_loss = self._scale_edge_pred_loss(epoch, edge_pred_loss)
                total_batch_edge_pred_loss += edge_pred_loss.item()

                step_loss = pred_loss + edge_pred_loss

                prev_node_pred = sliding_window[:, [-1]]
                pred = prev_node_pred + pred_diff
                prev_edge_pred = edge_sliding_window[:, [-1]]
                edge_pred = prev_edge_pred + edge_pred_diff

                if self.use_physics_loss:
                    global_loss, local_loss = self._get_physics_loss(epoch, pred, prev_node_pred,
                                                                     prev_edge_pred, batch,
                                                                     current_timestep=i)
                    total_batch_global_mass_loss += global_loss.item()
                    total_batch_local_mass_loss += local_loss.item()
                    step_loss = step_loss + global_loss + local_loss

                total_batch_loss = total_batch_loss + step_loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    next_sliding_window = torch.cat((sliding_window[:, 1:], pred), dim=1)
                    next_edge_sliding_window = torch.cat((edge_sliding_window[:, 1:], edge_pred), dim=1)

                    sliding_window = next_sliding_window
                    edge_sliding_window = next_edge_sliding_window

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()

            # Loss Updates
            total_losses = (total_batch_pred_loss, total_batch_edge_pred_loss, total_batch_global_mass_loss, total_batch_local_mass_loss)
            avg_losses = train_utils.divide_losses(total_losses, current_num_timesteps)
            avg_pred_loss, avg_edge_pred_loss, avg_global_mass_loss, avg_local_mass_loss = avg_losses
            self._add_epoch_loss_ratio_to_scaler(epoch, avg_pred_loss, avg_edge_pred_loss, avg_global_mass_loss, avg_local_mass_loss)

            running_pred_loss += avg_pred_loss
            running_edge_pred_loss += avg_edge_pred_loss
            running_global_mass_loss += avg_global_mass_loss
            running_local_mass_loss += avg_local_mass_loss

        running_loss = running_pred_loss + running_edge_pred_loss + running_global_mass_loss + running_local_mass_loss
        running_losses = (running_loss, running_pred_loss, running_edge_pred_loss, running_global_mass_loss, running_local_mass_loss)
        epoch_losses = train_utils.divide_losses(running_losses, len(self.dataloader))
        epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = epoch_losses

        return epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss

    def validate(self):
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

    def _compute_edge_loss(self, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y_edge[:, :, timestep]
        return self.edge_loss_func(edge_pred, label)

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        pred = NodeAutoregressiveTrainer._override_pred_bc(self, pred, batch, timestep)
        edge_pred = EdgeAutoregressiveTrainer._override_pred_bc(self, edge_pred, batch, timestep)
        return pred, edge_pred

# ========= Methods for scaling losses =========

    def _scale_edge_pred_loss(self, epoch: int, edge_pred_loss: Tensor) -> Tensor:
        scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        if epoch < self.num_epochs_dyn_loss:
            return scaled_edge_pred_loss
        return scaled_edge_pred_loss * self.edge_loss_weight

    def _add_epoch_loss_ratio_to_scaler(self, epoch: int, pred_loss: float, edge_pred_loss: float, global_mass_loss: float, local_mass_loss: float):
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
        NodeAutoregressiveTrainer._add_epoch_loss_ratio_to_scaler(self, epoch, pred_loss, global_mass_loss, local_mass_loss)

    def _update_loss_scaler_for_epoch(self, epoch: int):
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.update_scale_from_epoch()
            self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')
        NodeAutoregressiveTrainer._update_loss_scaler_for_epoch(self, epoch)
