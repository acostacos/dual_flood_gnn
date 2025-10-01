import os

from contextlib import redirect_stdout
from torch import Tensor
from testing import DualAutoregressiveTester
from typing import Callable
from utils import LossScaler, physics_utils, train_utils

from .node_regression_trainer import NodeRegressionTrainer
from .edge_regression_trainer import EdgeRegressionTrainer

class DualRegressionTrainer(NodeRegressionTrainer, EdgeRegressionTrainer):
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
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()

            running_pred_loss = 0.0
            running_edge_pred_loss = 0.0
            running_global_mass_loss = 0.0
            running_local_mass_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
                pred_diff, edge_pred_diff = self.model(x, edge_index, edge_attr)
                pred_diff, edge_pred_diff = self._override_pred_bc(pred_diff, edge_pred_diff, batch)

                pred_loss = self._compute_node_loss(pred_diff, batch)
                pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                running_pred_loss += pred_loss.item()

                edge_pred_loss = self._compute_edge_loss(edge_pred_diff, batch)
                edge_pred_loss = self._scale_edge_pred_loss(epoch, edge_pred_loss)
                running_edge_pred_loss += edge_pred_loss.item()

                loss = pred_loss + edge_pred_loss

                if self.use_physics_loss:
                    previous_timesteps = self.dataloader.dataset.previous_timesteps
                    curr_water_volume, curr_face_flow = physics_utils.get_physics_info_node_edge(x, edge_attr, previous_timesteps, batch)
                    pred = curr_water_volume + pred_diff
                    global_loss, local_loss = self._get_physics_loss(epoch, pred, curr_water_volume,
                                                                     curr_face_flow, pred_loss, batch)
                    running_global_mass_loss += global_loss.item()
                    running_local_mass_loss += local_loss.item()
                    loss = loss + global_loss + local_loss

                loss.backward()
                self.optimizer.step()

            running_loss = running_pred_loss + running_edge_pred_loss + running_global_mass_loss + running_local_mass_loss
            running_losses = (running_loss, running_pred_loss, running_edge_pred_loss, running_global_mass_loss, running_local_mass_loss)
            epoch_losses = train_utils.divide_losses(running_losses, len(self.dataloader))
            epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = epoch_losses

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tTotal Loss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if self.use_physics_loss:
                self._log_epoch_physics_loss(global_mass_epoch_loss, local_mass_epoch_loss)

            self._update_loss_scaler_for_epoch(epoch)

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

# ========= Methods for scaling losses =========

    def _scale_edge_pred_loss(self, epoch: int, basis_loss: Tensor, edge_pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(basis_loss, edge_pred_loss)
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        else:
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_loss_weight
        return scaled_edge_pred_loss

    def _update_loss_scaler_for_epoch(self, epoch: int):
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.update_scale_from_epoch()
            self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')
        NodeRegressionTrainer._update_loss_scaler_for_epoch(self, epoch)
