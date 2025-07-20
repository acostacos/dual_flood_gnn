import numpy as np
import torch
from data import AutoRegressiveDataLoader, FloodEventDataset

from .dual_regression_trainer import DualRegressionTrainer

class DualAutoRegressiveTrainer(DualRegressionTrainer):
    def __init__(self,
                 num_timesteps: int = 1,
                 curriculum_epochs: int = 10,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_timesteps = num_timesteps
        self.curriculum_epochs = curriculum_epochs

        # Get non-boundary nodes/edges and threshold for metric computation
        ds: FloodEventDataset = self.dataloader.dataset
        self.non_boundary_nodes_mask = ds.boundary_condition.get_non_boundary_nodes_mask()
        self.non_boundary_edges_mask = ds.boundary_condition.non_boundary_edges_mask

        # Get sliding window indices
        sliding_window_length = ds.previous_timesteps + 1
        target_nodes_idx = ds.DYNAMIC_NODE_FEATURES.index(ds.NODE_TARGET_FEATURE)
        self.start_target_idx = ds.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_target_idx = self.start_target_idx + sliding_window_length

        target_edges_idx = ds.DYNAMIC_EDGE_FEATURES.index(ds.EDGE_TARGET_FEATURE)
        self.start_target_edges_idx = ds.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_target_edges_idx = self.start_target_edges_idx + sliding_window_length

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = 1
        dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset, batch_size=self.batch_size, num_timesteps=current_num_timesteps)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            running_edge_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            group_losses = None
            sliding_window = None
            edge_sliding_window = None
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)

                if i % current_num_timesteps == 0:
                    self.optimizer.zero_grad()
                    group_losses = []
                    sliding_window = batch.x.clone()[:, self.start_target_idx:self.end_target_idx]
                    edge_sliding_window = batch.edge_attr.clone()[:, self.start_target_edges_idx:self.end_target_edges_idx]

                # Override graph data with sliding window
                # Only override non-boundary nodes to keep boundary conditions intact
                batch_non_boundary_nodes_mask = np.tile(self.non_boundary_nodes_mask, batch.num_graphs)
                batch.x[batch_non_boundary_nodes_mask, self.start_target_idx:self.end_target_idx] \
                    = sliding_window[batch_non_boundary_nodes_mask]

                # Only override non-boundary edges to keep boundary conditions intact
                batch_non_boundary_edges_mask = np.tile(self.non_boundary_edges_mask, batch.num_graphs)
                batch.edge_attr[batch_non_boundary_edges_mask, self.start_target_edges_idx:self.end_target_edges_idx] \
                    = edge_sliding_window[batch_non_boundary_edges_mask]

                pred, edge_pred = self.model(batch)

                sliding_window = torch.concat((sliding_window[:, 1:], pred.detach()), dim=1)
                edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred.detach()), dim=1)

                label = batch.y
                pred_loss = self.loss_func(pred, label)
                pred_loss =  pred_loss * self.pred_loss_percent
                running_pred_loss += pred_loss.item()

                edge_label = batch.y_edge
                edge_pred_loss = self.loss_func(edge_pred, edge_label)
                self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
                edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
                running_edge_pred_loss += edge_pred_loss.item()

                loss = pred_loss + edge_pred_loss

                if self.use_physics_loss:
                    physics_loss = self._get_epoch_physics_loss(pred, loss, batch)
                    loss = loss * self.pred_loss_percent + physics_loss

                group_losses.append(loss)

                if ((i + 1) % current_num_timesteps == 0) or (i + 1 == len(dataloader)):
                    torch.stack(group_losses).mean().backward()
                    self.optimizer.step()

            running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
            epoch_loss = running_loss / len(dataloader)
            pred_epoch_loss = running_pred_loss / len(dataloader)
            edge_pred_epoch_loss = running_edge_pred_loss / len(dataloader)

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if epoch < self.num_epochs_dyn_weight:
                self.edge_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

        self.training_stats.end_train()
