import os

from contextlib import redirect_stdout
from torch import Tensor
from testing import DualAutoregressiveTester
from typing import Callable
from utils import LossScaler

from .node_regression_trainer import NodeRegressionTrainer
from .edge_regression_trainer import EdgeRegressionTrainer

class DualRegressionTrainer(NodeRegressionTrainer, EdgeRegressionTrainer):
    def __init__(self,
                 edge_loss_func: Callable,
                 edge_pred_loss_scale: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edge_loss_func = edge_loss_func
        self.edge_loss_scaler = LossScaler(initial_scale=edge_pred_loss_scale)

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            running_edge_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
                pred, edge_pred = self.model(x, edge_index, edge_attr)
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch)

                pred_loss = self._compute_node_loss(pred, batch)
                running_pred_loss += pred_loss.item()

                edge_pred_loss = self._compute_edge_loss(edge_pred, batch)
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

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tTotal Loss: {epoch_loss:.4e}\n'
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

            if hasattr(self, 'early_stopping'):
                val_node_rmse, val_edge_rmse = self.validate()
                self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
                self.training_stats.log(f'\tValidation Edge RMSE: {val_edge_rmse:.4e}')
                self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
                self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

                if self.early_stopping((val_node_rmse, val_edge_rmse), self.model):
                    self.training_stats.log(f'Early stopping triggered at epoch {epoch + 1}.')
                    break

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

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

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch) -> Tensor:
        pred = NodeRegressionTrainer._override_pred_bc(self, pred, batch)
        edge_pred = EdgeRegressionTrainer._override_pred_bc(self, edge_pred, batch)
        return pred, edge_pred

    def _scale_edge_pred_loss(self, epoch: int, pred_loss: Tensor, edge_pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
        scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        return scaled_edge_pred_loss
