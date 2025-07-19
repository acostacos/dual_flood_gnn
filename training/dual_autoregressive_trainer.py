import numpy as np
import torch
from data import AutoRegressiveDataLoader
from utils import LossScaler

from .base_trainer import BaseTrainer

class DualAutoRegressiveTrainer(BaseTrainer):
    def __init__(self,
                 edge_pred_loss_percent: float = 0.5,
                 batch_size: int = 1,
                 num_timesteps: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataloader = AutoRegressiveDataLoader(self.dataloader.dataset,
                                                   batch_size=batch_size,
                                                   num_timesteps=num_timesteps)

        self.edge_pred_loss_percent = edge_pred_loss_percent
        self.num_timesteps = num_timesteps

        self.edge_loss_scaler = LossScaler()
        self.pred_loss_percent -= self.edge_pred_loss_percent

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            running_edge_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            # Get non-boundary nodes/edges and threshold for metric computation
            dataset = self.dataloader.dataset
            non_boundary_nodes_mask = dataset.boundary_condition.get_non_boundary_nodes_mask()
            non_boundary_edges_mask = dataset.boundary_condition.non_boundary_edges_mask

            # Get sliding window indices
            sliding_window_length = dataset.previous_timesteps + 1
            target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
            start_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
            end_target_idx = start_target_idx + sliding_window_length

            target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
            start_target_edges_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
            end_target_edges_idx = start_target_edges_idx + sliding_window_length

            sliding_window = None
            edge_sliding_window = None
            for i, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)

                if (i % self.num_timesteps == 0):
                    self.optimizer.zero_grad()

                    sliding_window = batch.x.clone()[:, start_target_idx:end_target_idx]
                    edge_sliding_window = batch.edge_attr.clone()[:, start_target_edges_idx:end_target_edges_idx]

                # Override graph data with sliding window
                # Only override non-boundary nodes to keep boundary conditions intact
                batch_non_boundary_nodes_mask = np.tile(non_boundary_nodes_mask, batch.num_graphs)
                batch.x[batch_non_boundary_nodes_mask, start_target_idx:end_target_idx] = sliding_window[batch_non_boundary_nodes_mask]

                # Only override non-boundary edges to keep boundary conditions intact
                batch_non_boundary_edges_mask = np.tile(non_boundary_edges_mask, batch.num_graphs)
                batch.edge_attr[batch_non_boundary_edges_mask, start_target_edges_idx:end_target_edges_idx] = edge_sliding_window[batch_non_boundary_edges_mask]

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

                loss = loss / self.num_timesteps
                loss.backward()

                if ((i + 1) % self.num_timesteps == 0) or (i + 1 == len(self.dataloader)):
                    self.optimizer.step()

            running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
            epoch_loss = running_loss / len(self.dataloader)
            pred_epoch_loss = running_pred_loss / len(self.dataloader)
            edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

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
