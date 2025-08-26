import os
import numpy as np

from contextlib import redirect_stdout
from data import FloodEventDataset
from testing import EdgeAutoregressiveTester
from torch import Tensor

from .base_trainer import BaseTrainer

class EdgeRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: FloodEventDataset = self.dataloader.dataset
        self.inflow_edges_mask = ds.boundary_condition.inflow_edges_mask

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.total_num_epochs):
            self.model.train()
            running_edge_pred_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                edge_pred = self.model(batch)
                edge_pred = self._override_pred_bc(edge_pred, batch)

                loss = self._compute_edge_loss(edge_pred, batch)
                running_edge_pred_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(edge_pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if hasattr(self, 'early_stopping'):
                val_edge_rmse = self.validate()
                self.training_stats.log(f'\n\tValidation Edge RMSE: {val_edge_rmse:.4e}')
                self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

                if self.early_stopping(val_edge_rmse, self.model):
                    self.training_stats.log(f'Early stopping triggered at epoch {epoch + 1}.')
                    break

        self.training_stats.end_train()

    def validate(self):
        val_tester = EdgeAutoregressiveTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        edge_rmse = val_tester.get_avg_edge_rmse()
        return edge_rmse

    def _compute_edge_loss(self, edge_pred: Tensor, batch) -> Tensor:
        label = batch.y_edge
        return self.loss_func(edge_pred, label)

    def _override_pred_bc(self, edge_pred: Tensor, batch) -> Tensor:
        # Only override inflow edges as outflow edges are predicted by the model
        batch_inflow_edges_mask = np.tile(self.inflow_edges_mask, batch.num_graphs)
        edge_pred[batch_inflow_edges_mask] = batch.y_edge[batch_inflow_edges_mask]
        return edge_pred
