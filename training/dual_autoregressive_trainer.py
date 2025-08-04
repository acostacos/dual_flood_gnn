import time
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
        self.non_boundary_nodes_mask = ~ds.boundary_condition.boundary_nodes_mask
        self.non_boundary_edges_mask = ~ds.boundary_condition.boundary_edges_mask

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
        for epoch in range(self.total_num_epochs):
            epoch_start_time = time.time()

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

                reset_autoregressive = i % current_num_timesteps == 0
                if reset_autoregressive:
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
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch)

                sliding_window = torch.concat((sliding_window[:, 1:], pred.detach()), dim=1)
                edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred.detach()), dim=1)

                label = batch.y
                pred_loss = self.loss_func(pred, label)
                pred_loss =  pred_loss * self.pred_loss_percent
                running_pred_loss += pred_loss.item()

                edge_label = batch.y_edge
                edge_pred_loss = self.loss_func(edge_pred, edge_label)
                edge_pred_loss = self._scale_edge_pred_loss(epoch, pred_loss, edge_pred_loss)
                running_edge_pred_loss += edge_pred_loss.item()

                loss = pred_loss + edge_pred_loss

                if self.use_physics_loss:
                    prev_edge_pred = None if reset_autoregressive else edge_sliding_window[:, [-2]]
                    physics_loss = self._get_epoch_physics_loss(epoch, pred, loss, batch, prev_edge_pred)
                    loss += physics_loss

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
            elif ((epoch - self.num_epochs_dyn_loss) != 0
                  and (epoch - self.num_epochs_dyn_loss) % self.curriculum_epochs == 0
                  and current_num_timesteps < self.num_timesteps):
                current_num_timesteps += 1
                dataloader = AutoRegressiveDataLoader(dataset=self.dataloader.dataset, batch_size=self.batch_size, num_timesteps=current_num_timesteps)
                self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps')

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.training_stats.log(f'\tEpoch Duration: {epoch_duration:.2f} seconds')

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    # def train(self):
    #     '''Pushforward Trick + Stability Loss'''
    #     # TODO: CURRENT BUG = stability loss is not computed correctly when it reaches the end of the event. Need to check for the last batch of each event.
    #     self.training_stats.start_train()
    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         running_pred_loss = 0.0
    #         running_edge_pred_loss = 0.0
    #         running_stability_loss = 0.0
    #         if self.use_physics_loss:
    #             self._reset_epoch_physics_running_loss()

    #         prev_pred = None
    #         prev_edge_pred = None
    #         for batch in self.dataloader:
    #             batch = batch.to(self.device)

    #             # One-step prediction
    #             pred, edge_pred = self.model(batch)

    #             label = batch.y
    #             pred_loss = self.loss_func(pred, label)
    #             pred_loss =  pred_loss * self.pred_loss_percent
    #             running_pred_loss += pred_loss.item()

    #             edge_label = batch.y_edge
    #             edge_pred_loss = self.loss_func(edge_pred, edge_label)
    #             self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
    #             edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
    #             running_edge_pred_loss += edge_pred_loss.item()

    #             one_step_loss = pred_loss + edge_pred_loss

    #             if prev_pred is not None and prev_edge_pred is not None:
    #                 # Stability Loss
    #                 # Prediction for timestep t
    #                 pred_stab, edge_pred_stab = self.model(batch)
    #                 d_pred_stab, d_edge_pred_stab = pred_stab.detach(), edge_pred_stab.detach()

    #                 # Prediction for timestep t+1
    #                 next_batch = Batch.from_data_list(batch.to_data_list()[1:]) # Get next batch (t+1) by slicing the batch

    #                 next_ts_node_mask = batch.batch != 0
    #                 next_batch.x[:, self.target_nodes_idx:self.target_nodes_idx+1] = \
    #                     d_pred_stab[next_ts_node_mask, :]
    #                 next_ts_edge_mask = torch.arange(batch.num_edges, device=self.device) >= (batch.num_edges // batch.num_graphs)
    #                 next_batch.edge_attr[:, self.target_edges_idx:self.target_edges_idx+1] = \
    #                     d_edge_pred_stab[next_ts_edge_mask, :]
    #                 pred_stab_next, edge_pred_stab_next = self.model(next_batch)

    #                 stab_pred_loss = self.loss_func(pred_stab_next, next_batch.y)
    #                 stab_pred_loss =  stab_pred_loss * self.pred_loss_percent

    #                 stab_edge_pred_loss = self.loss_func(edge_pred_stab_next, next_batch.y_edge)
    #                 stab_edge_pred_loss = self.edge_loss_scaler.scale_loss(stab_edge_pred_loss) * self.edge_pred_loss_percent

    #                 stability_loss = stab_pred_loss + stab_edge_pred_loss
    #                 running_stability_loss += stability_loss.item()

    #                 loss = one_step_loss + stability_loss

    #                 loss.backward()
    #                 self.optimizer.step()

    #         running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
    #         epoch_loss = running_loss / len(self.dataloader)
    #         pred_epoch_loss = running_pred_loss / len(self.dataloader)
    #         edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)
    #         epoch_stability_loss = running_stability_loss / len(self.dataloader)

    #         logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
    #         logging_str += f'\tLoss: {epoch_loss:.4e}\n'
    #         logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}\n'
    #         logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}\n'
    #         logging_str += f'\tStability Loss: {epoch_stability_loss:.4e}'
    #         self.training_stats.log(logging_str)

    #         self.training_stats.add_loss(epoch_loss)
    #         self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
    #         self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)
    #         self.training_stats.add_loss_component('stability_loss', epoch_stability_loss)

    #         if epoch < self.num_epochs_dyn_weight:
    #             self.edge_loss_scaler.update_scale_from_epoch()
    #             self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')

    #         if self.use_physics_loss:
    #             self._process_epoch_physics_loss(epoch)

    #     self.training_stats.end_train()
