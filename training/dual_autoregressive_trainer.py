import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from testing import DualAutoregressiveTester
from typing import Tuple
from utils import EarlyStopping, LossScaler

from .node_autoregressive_trainer import NodeAutoregressiveTrainer
from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer

class DualAutoregressiveTrainer(NodeAutoregressiveTrainer, EdgeAutoregressiveTrainer):
    def __init__(self,
                 edge_pred_loss_scale: float = 1.0,
                 edge_pred_loss_percent: float = 0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edge_pred_loss_percent = edge_pred_loss_percent

        self.edge_loss_scaler = LossScaler(initial_scale=edge_pred_loss_scale)
        self.pred_loss_percent -= self.edge_pred_loss_percent

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps

        for epoch in range(self.total_num_epochs):
            train_start_time = time.time()

            epoch_loss, pred_epoch_loss, edge_pred_epoch_loss = self._train_model(epoch, current_num_timesteps)

            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
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

            val_node_rmse, val_edge_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.log(f'\tValidation Edge RMSE: {val_edge_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
            self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

            # Previous critera to increase autoregression = non_dyn_epoch_num != 0 and non_dyn_epoch_num % self.curriculum_epochs == 0
            if self.early_stopping((val_node_rmse, val_edge_rmse), self.model):
                self.training_stats.log(f'\tEarly stopping triggered after {non_dyn_epoch_num + 1} epochs.')

                if current_num_timesteps < self.total_num_timesteps:
                    current_num_timesteps += 1
                    self.early_stopping = EarlyStopping(patience=self.early_stopping.patience)
                    self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps.')
                    self.lr_scheduler.step()
                    self.training_stats.log(f'\tDecayed learning rate to {self.lr_scheduler.get_last_lr()[0]:.4e}.')
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
            x, edge_attr, edge_index = batch.x[:, :, 0], batch.edge_attr[:, :, 0], batch.edge_index

            total_batch_loss = 0.0
            sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            edge_sliding_window = edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
            for i in range(current_num_timesteps):
                x, edge_attr = batch.x[:, :, i], batch.edge_attr[:, :, i]

                # Override graph data with sliding window
                x = torch.concat([x[:, :self.start_node_target_idx], sliding_window, x[:, self.end_node_target_idx:]], dim=1)
                edge_attr = torch.concat([edge_attr[:, :self.start_edge_target_idx], edge_sliding_window, edge_attr[:, self.end_edge_target_idx:]], dim=1)

                pred, edge_pred = self.model(x, edge_index, edge_attr)
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch, i)

                pred_loss = self._compute_node_loss(pred, batch, i)
                pred_loss = pred_loss * self.pred_loss_percent
                running_pred_loss += pred_loss.item()

                edge_pred_loss = self._compute_edge_loss(edge_pred, batch, i)
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

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()

        running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
        epoch_loss = running_loss / len(self.dataloader)
        pred_epoch_loss = running_pred_loss / len(self.dataloader)
        edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

        return epoch_loss, pred_epoch_loss, edge_pred_epoch_loss

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

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        pred = NodeAutoregressiveTrainer._override_pred_bc(self, pred, batch, timestep)
        edge_pred = EdgeAutoregressiveTrainer._override_pred_bc(self, edge_pred, batch, timestep)
        return pred, edge_pred

    def _scale_edge_pred_loss(self, epoch: int, pred_loss: Tensor, edge_pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        else:
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
        return scaled_edge_pred_loss
