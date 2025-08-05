import math
import torch

from loss import GlobalMassConservationLoss, LocalMassConservationLoss
from data import FloodEventDataset
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from typing import Callable, Optional
from utils import LossScaler, Logger
from utils.training_stats import TrainingStats


class BaseTrainer:
    def __init__(self,
                 model: Module,
                 dataset: FloodEventDataset,
                 optimizer: Optimizer,
                 loss_func: Callable,
                 use_global_loss: bool = False,
                 global_mass_loss_scale: float = 1.0,
                 global_mass_loss_percent: float = 0.1,
                 use_local_loss: bool = False,
                 local_mass_loss_scale: float = 1.0,
                 local_mass_loss_percent: float = 0.1,
                 delta_t: int = 30,
                 batch_size: int = 64,
                 num_epochs: int = 100,
                 num_epochs_dyn_loss: int = 10,
                 gradient_clip_value: Optional[float] = None,
                 logger: Logger = None,
                 device: str = 'cpu'):
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.use_physics_loss = use_global_loss or use_local_loss
        self.use_global_loss = use_global_loss
        self.global_mass_loss_percent = global_mass_loss_percent
        self.use_local_loss = use_local_loss
        self.local_mass_loss_percent = local_mass_loss_percent
        self.delta_t = delta_t
        self.batch_size = batch_size
        self.num_epochs_dyn_loss = num_epochs_dyn_loss
        self.gradient_clip_value = gradient_clip_value
        self.device = device
        self.training_stats = TrainingStats(logger=logger)

        self.total_num_epochs = num_epochs_dyn_loss + num_epochs
        self.training_stats.log(f'Using dynamic loss weight adjustment for the first {self.num_epochs_dyn_loss}/{self.total_num_epochs} epochs')

        self.pred_loss_percent = 1.0
        if self.use_global_loss:
            self.global_loss_func = GlobalMassConservationLoss(
                previous_timesteps=dataset.previous_timesteps,
                normalizer=dataset.normalizer,
                is_normalized=dataset.is_normalized,
                delta_t=self.delta_t
            )
            self.global_loss_scaler = LossScaler(initial_scale=global_mass_loss_scale)
            self.pred_loss_percent -= global_mass_loss_percent

        if self.use_local_loss:
            self.local_loss_func = LocalMassConservationLoss(
                previous_timesteps=dataset.previous_timesteps,
                normalizer=dataset.normalizer,
                is_normalized=dataset.is_normalized,
                delta_t=self.delta_t,
            )
            self.local_loss_scaler = LossScaler(initial_scale=local_mass_loss_scale)
            self.pred_loss_percent -= local_mass_loss_percent

    def _reset_epoch_physics_running_loss(self):
        if self.use_global_loss:
            self.running_orig_global_physics_loss = 0.0
            self.running_global_physics_loss = 0.0
        if self.use_local_loss:
            self.running_orig_local_physics_loss = 0.0
            self.running_local_physics_loss = 0.0

    def _get_epoch_physics_loss(self, epoch, pred: Tensor, pred_loss: Tensor, batch, prev_edge_pred: Tensor = None) -> Tensor:
        if not self.use_global_loss and not self.use_local_loss:
            raise ValueError("At least one of global or local physics loss must be enabled.")

        if self.use_global_loss:
            global_physics_loss = self._get_epoch_global_mass_loss(epoch, pred, pred_loss, batch, prev_edge_pred)

        if not self.use_local_loss:
            return global_physics_loss

        if self.use_local_loss:
            local_physics_loss = self._get_epoch_local_mass_loss(epoch, pred, pred_loss, batch, prev_edge_pred)

        if not self.use_global_loss:
            return local_physics_loss

        return local_physics_loss + global_physics_loss

    def _get_epoch_global_mass_loss(self, epoch: int, pred: Tensor, pred_loss: Tensor, batch, prev_edge_pred: Tensor = None) -> Tensor:
        if prev_edge_pred is None:
            # If previous edge prediction is not provided, use the face flow from batch
            prev_edge_pred = batch.global_mass_info['face_flow']

        global_physics_loss = self.global_loss_func(pred, prev_edge_pred, batch)
        self.running_orig_global_physics_loss += global_physics_loss.item()

        scaled_global_physics_loss = self._scale_global_mass_loss(epoch, pred_loss, global_physics_loss)
        self.running_global_physics_loss += scaled_global_physics_loss.item()
        return scaled_global_physics_loss

    def _scale_global_mass_loss(self, epoch: int, pred_loss: Tensor, global_mass_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.global_loss_scaler.add_epoch_loss_ratio(pred_loss, global_mass_loss)
            scaled_global_physics_loss = self.global_loss_scaler.scale_loss(global_mass_loss)
        else:
            scaled_global_physics_loss = self.global_loss_scaler.scale_loss(global_mass_loss) * self.global_mass_loss_percent
        return scaled_global_physics_loss

    def _get_epoch_local_mass_loss(self, epoch: int, pred: Tensor, pred_loss: Tensor, batch, prev_edge_pred: Tensor = None) -> Tensor:
        if prev_edge_pred is None:
            # If previous edge prediction is not provided, use the face flow from batch
            prev_edge_pred = batch.local_mass_info['face_flow']

        local_physics_loss = self.local_loss_func(pred, prev_edge_pred, batch)
        self.running_orig_local_physics_loss += local_physics_loss.item()

        scaled_local_physics_loss = self._scale_local_mass_loss(epoch, pred_loss, local_physics_loss)
        self.running_local_physics_loss += scaled_local_physics_loss.item()
        return scaled_local_physics_loss

    def _scale_local_mass_loss(self, epoch: int, pred_loss: Tensor, local_mass_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.local_loss_scaler.add_epoch_loss_ratio(pred_loss, local_mass_loss)
            scaled_local_physics_loss = self.local_loss_scaler.scale_loss(local_mass_loss)
        else:
            scaled_local_physics_loss = self.local_loss_scaler.scale_loss(local_mass_loss) * self.local_mass_loss_percent
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

            orig_global_physics_epoch_loss = self.running_orig_global_physics_loss / len(self.dataloader)
            self.training_stats.add_loss_component('orig_global_physics_loss', orig_global_physics_epoch_loss)

            if epoch < self.num_epochs_dyn_loss:
                self.global_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Global Mass Loss Weight to {self.global_loss_scaler.scale:.4e}')

        if self.use_local_loss:
            local_physics_epoch_loss = self.running_local_physics_loss / len(self.dataloader)
            self.training_stats.log(f'\tLocal Physics Loss: {local_physics_epoch_loss:.4e}')
            self.training_stats.add_loss_component('local_physics_loss', local_physics_epoch_loss)

            orig_local_physics_epoch_loss = self.running_orig_local_physics_loss / len(self.dataloader)
            self.training_stats.add_loss_component('orig_local_physics_loss', orig_local_physics_epoch_loss)

            if epoch < self.num_epochs_dyn_loss:
                self.local_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Local Mass Loss Weight to {self.local_loss_scaler.scale:.4e}')

    def _add_scaled_physics_loss_history(self):
        if self.use_global_loss:
            self.training_stats.add_additional_info('global_scaled_loss_ratios', self.global_loss_scaler.scaled_loss_ratio_history)

        if self.use_local_loss:
            self.training_stats.add_additional_info('local_scaled_loss_ratios', self.local_loss_scaler.scaled_loss_ratio_history)

    def train(self):
        raise NotImplementedError("Train method not implemented.")

    def print_stats_summary(self):
        self.training_stats.print_stats_summary()

    def save_stats(self, filepath: str):
        self.training_stats.save_stats(filepath)

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)
        self.training_stats.log(f'Saved model to: {model_path}')
