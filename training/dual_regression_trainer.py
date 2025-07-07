from utils import LossScaler

from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self,
                 edge_pred_loss_percent: float = 0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edge_pred_loss_percent = edge_pred_loss_percent
        self.edge_loss_scaler = LossScaler()

    def train(self):
        # TODO: How to handle percentage between node ande edge?
        self.pred_loss_percent -= self.pred_loss_percent * self.edge_pred_loss_percent
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
                pred, edge_pred = self.model(batch)

                label = batch.y
                pred_loss = self.loss_func(pred, label)
                pred_loss =  pred_loss * self.pred_loss_percent
                running_pred_loss += pred_loss.item()

                edge_label = batch.y_edge
                edge_pred_loss = self.loss_func(edge_pred, edge_label)
                self.edge_loss_scaler.add_epoch_loss_ratio(pred_loss, edge_pred_loss)
                scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_pred_loss_percent
                running_edge_pred_loss += scaled_edge_pred_loss.item()

                loss = pred_loss + edge_pred_loss

                if self.use_physics_loss:
                    physics_loss = self._get_epoch_physics_loss(pred, loss, batch)
                    loss = loss * self.pred_loss_percent + physics_loss

                loss.backward()
                self.optimizer.step()

            running_loss = running_pred_loss + running_edge_pred_loss
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
