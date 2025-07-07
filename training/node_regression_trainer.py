from .base_trainer import BaseTrainer

class NodeRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_pred_loss = 0.0
            if self.use_physics_loss:
                self._reset_epoch_physics_running_loss()

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                pred = self.model(batch)

                label = batch.y
                loss = self.loss_func(pred, label)
                running_pred_loss += loss.item()

                if self.use_physics_loss:
                    physics_loss = self._get_epoch_physics_loss(pred, loss, batch)
                    loss = loss * self.pred_loss_percent + physics_loss

                loss.backward()
                self.optimizer.step()

            pred_epoch_loss = running_pred_loss / len(self.dataloader)

            epoch_loss = self._get_epoch_total_running_loss(running_pred_loss) / len(self.dataloader)
            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tPrediction Loss: {pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)

            if self.use_physics_loss:
                self._process_epoch_physics_loss(epoch)

        self.training_stats.end_train()
