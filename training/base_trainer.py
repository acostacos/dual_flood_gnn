import math
import torch

from loss import global_mass_conservation_loss, local_mass_conservation_loss
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from typing import Callable
from utils import TrainingStats, LossScaler, Logger


class BaseTrainer:
    def __init__(self,
                 model: Module,
                 dataloader: DataLoader,
                 optimizer: Optimizer,
                 loss_func: Callable,
                 use_global_loss: bool = False,
                 global_mass_loss_percent: float = 0.1,
                 use_local_loss: bool = False,
                 local_mass_loss_percent: float = 0.1,
                 delta_t: int = 30,
                 logger: Logger = None,
                 num_epochs: int = 100,
                 device: str = 'cpu'):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.use_physics_loss = use_global_loss or use_local_loss
        self.use_global_loss = use_global_loss
        self.global_mass_loss_percent = global_mass_loss_percent
        self.use_local_loss = use_local_loss
        self.local_mass_loss_percent = local_mass_loss_percent
        self.delta_t = delta_t
        self.num_epochs = num_epochs
        self.device = device

        self.training_stats = TrainingStats(logger=logger)
        self.is_normalized = dataloader.dataset.is_normalized
        self.normalizer = dataloader.dataset.normalizer
        self.boundary_condition = dataloader.dataset.boundary_condition

        self.num_epochs_dyn_weight = self._get_num_epochs_dynamic_loss_weight(num_epochs)
        self.training_stats.log(f'Using dynamic loss weight adjustment for the first {self.num_epochs_dyn_weight} epochs')

        self.pred_loss_percent = 1.0
        if self.use_global_loss:
            self.global_loss_scaler = LossScaler()
            self.pred_loss_percent -= global_mass_loss_percent

        if self.use_local_loss:
            self.local_loss_scaler = LossScaler()
            self.pred_loss_percent -= local_mass_loss_percent

    def _get_num_epochs_dynamic_loss_weight(self, num_epochs: int) -> int:
        return int(math.ceil(num_epochs * 0.1))

    def _reset_epoch_physics_running_loss(self):
        if self.use_global_loss:
            self.running_global_physics_loss = 0.0
        if self.use_local_loss:
            self.running_local_physics_loss = 0.0

    def _get_epoch_physics_loss(self, pred: Tensor, pred_loss: Tensor, batch) -> Tensor:
        physics_loss = torch.zeros(1, device=self.device)
        if self.use_global_loss:
            global_physics_loss = self._get_epoch_global_mass_loss(pred, pred_loss, batch)
            physics_loss += global_physics_loss

        if self.use_local_loss:
            local_physics_loss = self._get_epoch_local_mass_loss(pred, pred_loss, batch)
            physics_loss += local_physics_loss

        return physics_loss

    def _get_epoch_global_mass_loss(self, pred: Tensor, pred_loss: Tensor, batch) -> Tensor:
        global_physics_loss = global_mass_conservation_loss(pred, None, batch,
                                                            self.normalizer, self.boundary_condition,
                                                            is_normalized=self.is_normalized, delta_t=self.delta_t)
        self.global_loss_scaler.add_epoch_loss_ratio(pred_loss, global_physics_loss)

        scaled_global_physics_loss = self.global_loss_scaler.scale_loss(global_physics_loss) * self.global_mass_loss_percent
        self.running_global_physics_loss += scaled_global_physics_loss.item()
        return scaled_global_physics_loss

    def _get_epoch_local_mass_loss(self, pred: Tensor, pred_loss: Tensor, batch) -> Tensor:
        local_physics_loss = local_mass_conservation_loss(pred, None, batch,
                                                          self.normalizer, self.boundary_condition,
                                                          is_normalized=self.is_normalized, delta_t=self.delta_t)
        self.local_loss_scaler.add_epoch_loss_ratio(pred_loss, local_physics_loss)

        scaled_local_physics_loss = self.local_loss_scaler.scale_loss(local_physics_loss) * self.local_mass_loss_percent
        self.running_local_physics_loss += scaled_local_physics_loss.item()
        return scaled_local_physics_loss
    
    def _get_epoch_total_running_loss(self, current_running_loss: float) -> float:
        total_loss = current_running_loss
        if self.use_global_loss:
            total_loss += self.running_global_physics_loss
        if self.use_local_loss:
            total_loss += self.running_local_physics_loss
        return total_loss

    def _process_epoch_physics_loss(self, epoch: int):
        if self.use_global_loss:
            global_physics_epoch_loss = self.running_global_physics_loss / len(self.dataloader)
            self.training_stats.log(f'\tGlobal Physics Loss: {global_physics_epoch_loss:.4e}')
            self.training_stats.add_loss_component('global_physics_loss', global_physics_epoch_loss)

            if epoch < self.num_epochs_dyn_weight:
                self.global_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Global Mass Loss Weight to {self.global_loss_scaler.scale:.4e}')

        if self.use_local_loss:
            local_physics_epoch_loss = self.running_local_physics_loss / len(self.dataloader)
            self.training_stats.log(f'\tLocal Physics Loss: {local_physics_epoch_loss:.4e}')
            self.training_stats.add_loss_component('local_physics_loss', local_physics_epoch_loss)

            if epoch < self.num_epochs_dyn_weight:
                self.local_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Local Mass Loss Weight to {self.local_loss_scaler.scale:.4e}')

    def train(self):
        raise NotImplementedError("Train method not implemented.")

    def print_stats_summary(self):
        self.training_stats.print_stats_summary()

    def save_stats(self, filepath: str):
        self.training_stats.save_stats(filepath)

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)
        self.training_stats.log(f'Saved model to: {model_path}')
