import torch

from data import FloodEventDataset
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from typing import Callable, Optional
from utils import Logger, EarlyStopping
from utils.training_stats import TrainingStats

class BaseTrainer:
    def __init__(self,
                 model: Module,
                 dataset: FloodEventDataset,
                 optimizer: Optimizer,
                 loss_func: Callable,
                 batch_size: int = 64,
                 num_epochs: int = 100,
                 num_epochs_dyn_loss: int = 10,
                 gradient_clip_value: Optional[float] = None,
                 early_stopping_patience: Optional[int] = None,
                 val_dataset: Optional[FloodEventDataset] = None,
                 logger: Logger = None,
                 device: str = 'cpu'):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_epochs_dyn_loss = num_epochs_dyn_loss
        self.gradient_clip_value = gradient_clip_value
        self.val_dataset = val_dataset
        self.device = device

        if early_stopping_patience is not None:
            assert self.val_dataset is not None, "Validation dataset must be provided if early stopping is used."
            self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.training_stats = TrainingStats(logger=logger)

        assert self.num_epochs_dyn_loss <= self.num_epochs, "Number of epochs for dynamic loss scaling must not exceed total number of epochs."

    def train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def validate(self):
        raise NotImplementedError("Subclasses should implement this method if using early stopping.")

    def _clip_gradients(self):
        if self.gradient_clip_value is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.gradient_clip_value)

    def print_stats_summary(self):
        self.training_stats.print_stats_summary()

    def save_stats(self, filepath: str):
        self.training_stats.save_stats(filepath)

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)
        self.training_stats.log(f'Saved model to: {model_path}')
