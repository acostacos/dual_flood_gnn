import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from data import AutoregressiveFloodDataset
from testing import NodeAutoregressiveTester
from typing import Tuple

from .base_autoregressive_trainer import BaseAutoregressiveTrainer
from .physics_informed_trainer import PhysicsInformedTrainer

class NodeAutoregressiveTrainer(BaseAutoregressiveTrainer, PhysicsInformedTrainer):
    def __init__(self,
                 teacher_forcing_ratio: float = 0.95,
                 use_scheduled_sampling: bool = True,
                 *args, **kwargs):
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

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_scheduled_sampling = use_scheduled_sampling

        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps

        for epoch in range(self.total_num_epochs):
            train_start_time = time.time()

            epoch_loss, pred_epoch_loss = self._train_model(epoch, current_num_timesteps)


            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
            if self.use_scheduled_sampling:
                current_tf_ratio = self._scheduled_teacher_forcing_ratio(epoch, self.total_num_epochs)
                logging_str += f'\tTeacher Forcing Ratio: {current_tf_ratio:.3f}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)

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

            self.lr_scheduler.step()

            # Previous critera to increase autoregression = non_dyn_epoch_num != 0 and non_dyn_epoch_num % self.curriculum_epochs == 0
            if self.early_stopping(val_node_rmse, self.model):
                self.training_stats.log(f'\tEarly stopping triggered after {non_dyn_epoch_num + 1} epochs.')
                break

        self.training_stats.end_train()
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float]:
        self.model.train()
        running_pred_loss = 0.0
        if self.use_physics_loss:
            self._reset_epoch_physics_running_loss()

        # Get current teacher forcing ratio (optionally decay over epochs)
        current_teacher_forcing_ratio = self._scheduled_teacher_forcing_ratio(epoch, self.total_num_epochs)

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            x, edge_index = batch.x[:, :, 0], batch.edge_index

            total_batch_loss = 0.0
            sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            for i in range(current_num_timesteps):
                x, edge_attr = batch.x[:, :, i], batch.edge_attr[:, :, i]

                # Override graph data with sliding window
                x = torch.concat([x[:, :self.start_node_target_idx], sliding_window, x[:, self.end_node_target_idx:]], dim=1)

                pred = self.model(x, edge_index, edge_attr)
                pred = self._override_pred_bc(pred, batch, i)

                pred_loss = self._compute_node_loss(pred, batch, i)
                running_pred_loss += pred_loss.item()

                step_loss = pred_loss

                if self.use_physics_loss:
                    physics_loss = self._get_epoch_physics_loss(epoch, pred, pred_loss, batch, None)
                    step_loss = step_loss + physics_loss

                total_batch_loss = total_batch_loss + step_loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    use_teacher_forcing = torch.rand(1).item() < current_teacher_forcing_ratio
                    if use_teacher_forcing:
                        # Use ground truth for next timestep (teacher forcing)
                        ground_truth = batch.y[:, :, i]
                        next_sliding_window = torch.cat((sliding_window[:, 1:], ground_truth.detach()), dim=1)
                    else:
                        # Use model prediction for next timestep (scheduled sampling)
                        next_sliding_window = torch.cat((sliding_window[:, 1:], pred.detach()), dim=1)

                    sliding_window = next_sliding_window

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()

        running_loss = self._get_epoch_total_running_loss(running_pred_loss)
        epoch_loss = running_loss / len(self.dataloader)
        pred_epoch_loss = running_pred_loss / len(self.dataloader)

        return epoch_loss, pred_epoch_loss

    def _scheduled_teacher_forcing_ratio(self, epoch, total_epochs):
        if hasattr(self, 'use_scheduled_sampling') and not self.use_scheduled_sampling:
            return self.teacher_forcing_ratio

        min_ratio = 0.1  # Minimum teacher forcing ratio
        decay_rate = 0.95 # Decay rate per epoch after warm-up
        
        progress = epoch / max(1, total_epochs - 1)  # Avoid division by zero
        # Exponential decay
        # current_ratio = self.teacher_forcing_ratio * (decay_rate ** (progress * total_epochs))
        # Linear decay from teacher_forcing_ratio to min_ratio
        # current_ratio = self.teacher_forcing_ratio - (self.teacher_forcing_ratio - min_ratio) * progress
        current_ratio = self.teacher_forcing_ratio - (0.02 * epoch)

        # Ensure we don't go below minimum ratio
        return max(min_ratio, current_ratio)

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
