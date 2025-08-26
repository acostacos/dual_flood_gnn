import os
import numpy as np

from contextlib import redirect_stdout
from data import FloodEventDataset
from torch import Tensor
from testing import NodeAutoregressiveTester

from .physics_informed_trainer import PhysicsInformedTrainer

class NodeRegressionTrainer(PhysicsInformedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: FloodEventDataset = self.dataloader.dataset
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.total_num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                pred = self.model(batch)
                pred = self._override_pred_bc(pred, batch)

                loss = self._compute_node_loss(pred, batch)
                running_pred_loss += loss.item()

                if self.use_physics_loss:
                    physics_loss = self._get_epoch_physics_loss(epoch, pred, loss, batch)
                    loss = loss * self.pred_loss_percent + physics_loss

                loss.backward()
                self.optimizer.step()

            pred_epoch_loss = running_pred_loss / len(self.dataloader)

            epoch_loss = self._get_epoch_total_running_loss(running_pred_loss) / len(self.dataloader)
            logging_str = f'Epoch [{epoch + 1}/{self.total_num_epochs}]\n'
            logging_str += f'\tTotal Loss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

            if hasattr(self, 'early_stopping'):
                val_node_rmse = self.validate()
                self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
                self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)

                if self.early_stopping(val_node_rmse, self.model):
                    self.training_stats.log(f'Early stopping triggered at epoch {epoch + 1}.')
                    break

        self.training_stats.end_train()
        self._add_scaled_physics_loss_history()

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

    def _compute_node_loss(self, pred: Tensor, batch) -> Tensor:
        label = batch.y
        return self.loss_func(pred, label)

    def _override_pred_bc(self, pred: Tensor, batch) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask]
        return pred
