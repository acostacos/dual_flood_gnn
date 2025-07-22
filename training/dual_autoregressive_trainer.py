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
        # To check if batch size is valid
        self.dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset,
                                                   batch_size=self.batch_size,
                                                   num_timesteps=num_timesteps)

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

                    # TODO: See if you need this
                    # # Gradient clipping (avoid exploding gradients)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

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
            elif ((epoch - self.num_epochs_dyn_weight) != 0
                  and (epoch - self.num_epochs_dyn_weight) % self.curriculum_epochs == 0
                  and current_num_timesteps < self.num_timesteps):
                current_num_timesteps += 1
                dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset, batch_size=self.batch_size, num_timesteps=current_num_timesteps)
                self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps')

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

        self.training_stats.end_train()

    # TODO: Add pushforward method if needed
    # def train(self):
    #     '''Stability Loss'''
    #     self.training_stats.start_train()
    #     dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset, batch_size=self.batch_size, num_timesteps=interval)
    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         running_pred_loss = 0.0
    #         running_edge_pred_loss = 0.0
    #         if self.use_physics_loss:
    #             self._reset_epoch_physics_running_loss()

    #         group_losses = None
    #         sliding_window = None
    #         edge_sliding_window = None
    #         for i, batch in enumerate(dataloader):
    #             batch = batch.to(self.device)

    #             if (i % interval == 0):
    #                 self.optimizer.zero_grad()
    #                 group_losses = []
    #                 sliding_window = batch.x.clone()[:, self.start_target_idx:self.end_target_idx]
    #                 edge_sliding_window = batch.edge_attr.clone()[:, self.start_target_edges_idx:self.end_target_edges_idx]

    #             # Override graph data with sliding window
    #             # Only override non-boundary nodes to keep boundary conditions intact
    #             batch_non_boundary_nodes_mask = np.tile(self.non_boundary_nodes_mask, batch.num_graphs)
    #             batch.x[batch_non_boundary_nodes_mask, self.start_target_idx:self.end_target_idx] \
    #                 = sliding_window[batch_non_boundary_nodes_mask]

    #             # Only override non-boundary edges to keep boundary conditions intact
    #             batch_non_boundary_edges_mask = np.tile(self.non_boundary_edges_mask, batch.num_graphs)
    #             batch.edge_attr[batch_non_boundary_edges_mask, self.start_target_edges_idx:self.end_target_edges_idx] \
    #                 = edge_sliding_window[batch_non_boundary_edges_mask]

    #             pred, edge_pred = self.model(batch)

    #             sliding_window = torch.concat((sliding_window[:, 1:], pred.detach()), dim=1)
    #             edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred.detach()), dim=1)

    #             label = batch.y
    #             pred_loss = self.loss_func(pred, label)
    #             pred_loss =  pred_loss * self.pred_loss_percent
    #             running_pred_loss += pred_loss.item()

    #             edge_label = batch.y_edge
    #             edge_pred_loss = self.loss_func(edge_pred, edge_label)
    #             self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
    #             edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
    #             running_edge_pred_loss += edge_pred_loss.item()

    #             loss = pred_loss + edge_pred_loss





    #             X = graph.ndata["x"]
    #             n_static = 12  # assumed static features dimension
    #             n_time = (X.shape[1] - n_static) // 2
    #             static_part = X[:, :n_static]
    #             water_depth_full = X[:, n_static:n_static + n_time]
    #             volume_full = X[:, n_static + n_time:n_static + 2 * n_time]
    #             # For one-step prediction, use dynamic features from indices 1: (last n_time_steps)
    #             water_depth_window_one = water_depth_full[:, 1:]
    #             volume_window_one = volume_full[:, 1:]
    #             X_one = torch.cat([static_part, water_depth_window_one, volume_window_one], dim=1)
    #             pred_one = self.model(X_one, graph.edata["x"], graph)
    #             one_step_loss = self.criterion(pred_one, graph.ndata["y"])

    #             # Stability branch (example implementation)
    #             water_depth_window_stab = water_depth_full[:, :n_time - 1]
    #             volume_window_stab = volume_full[:, :n_time - 1]
    #             X_stab = torch.cat([static_part, water_depth_window_stab, volume_window_stab], dim=1)
    #             pred_stab = self.model(X_stab, graph.edata["x"], graph)
    #             pred_stab_detached = pred_stab.detach()
    #             water_depth_updated = torch.cat(
    #                 [water_depth_full[:, 1:2], water_depth_full[:, 1:2] + pred_stab_detached[:, 0:1]],
    #                 dim=1
    #             )
    #             volume_updated = torch.cat(
    #                 [volume_full[:, 1:2], volume_full[:, 1:2] + pred_stab_detached[:, 1:2]],
    #                 dim=1
    #             )
    #             X_stab_updated = torch.cat([static_part, water_depth_updated, volume_updated], dim=1)
    #             pred_stab2 = self.model(X_stab_updated, graph.edata["x"], graph)
    #             stability_loss = self.criterion(pred_stab2, graph.ndata["y"])

    #             loss = one_step_loss + stability_loss
    #             loss_dict = {
    #                 "total_loss": loss,
    #                 "loss_one": one_step_loss,
    #                 "loss_stability": stability_loss
    #             }
    #             if self.use_physics_loss and physics_data is not None:
    #                 phy_loss = compute_physics_loss(pred_one, physics_data, graph, delta_t=self.delta_t)
    #                 loss = loss + self.physics_loss_weight * phy_loss
    #                 loss_dict["physics_loss"] = phy_loss

    #             group_losses.append(loss)

    #             if ((i + 1) % interval == 0) or (i + 1 == len(dataloader)):
    #                 torch.stack(group_losses).mean().backward()
    #                 self.optimizer.step()

    #         running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
    #         epoch_loss = running_loss / len(dataloader)
    #         pred_epoch_loss = running_pred_loss / len(dataloader)
    #         edge_pred_epoch_loss = running_edge_pred_loss / len(dataloader)

    #         logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
    #         logging_str += f'\tLoss: {epoch_loss:.4e}\n'
    #         logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}\n'
    #         logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
    #         self.training_stats.log(logging_str)

    #         self.training_stats.add_loss(epoch_loss)
    #         self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
    #         self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

    #         if epoch < self.num_epochs_dyn_weight:
    #             self.edge_loss_scaler.update_scale_from_epoch()
    #             self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')
    #         elif(epoch - self.num_epochs_dyn_weight) != 0 and (epoch - self.num_epochs_dyn_weight) % 15 == 0 and interval < self.num_timesteps:
    #             interval += 1
    #             dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset, batch_size=self.batch_size, num_timesteps=interval)
    #             self.training_stats.log(f'\tIncreased interval to {interval} timesteps')


    #         if self.use_physics_loss:
    #             self._process_epoch_physics_loss(epoch)

    #     self.training_stats.end_train()
