import numpy as np

from data import FloodEventDataset
from torch import Tensor
from utils import LossScaler

from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self,
                 edge_pred_loss_scale: float = 1.0,
                 edge_pred_loss_percent: float = 0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: FloodEventDataset = self.dataloader.dataset
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask
        self.inflow_edges_mask = ds.boundary_condition.inflow_edges_mask

        self.edge_pred_loss_percent = edge_pred_loss_percent

        self.edge_loss_scaler = LossScaler(initial_scale=edge_pred_loss_scale)
        self.pred_loss_percent -= self.edge_pred_loss_percent

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.total_num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            running_edge_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                pred, edge_pred = self.model(batch)
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch)

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
                    physics_loss = self._get_epoch_physics_loss(epoch, pred, pred_loss, batch)
                    loss = loss + physics_loss

                loss.backward()
                self.optimizer.step()

            running_loss = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))
            epoch_loss = running_loss / len(self.dataloader)
            pred_epoch_loss = running_pred_loss / len(self.dataloader)
            edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

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

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask]

        # Only override inflow edges as outflow edges are predicted by the model
        batch_inflow_edges_mask = np.tile(self.inflow_edges_mask, batch.num_graphs)
        edge_pred[batch_inflow_edges_mask] = batch.y_edge[batch_inflow_edges_mask]
        return pred, edge_pred

    def _scale_edge_pred_loss(self, epoch: int, pred_loss: Tensor, edge_pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        else:
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
        return scaled_edge_pred_loss
