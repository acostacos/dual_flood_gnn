import torch

from loss import GlobalMassConservationLoss, LocalMassConservationLoss
from data import FloodEventDataset
from torch import Tensor
from utils import LossScaler, physics_utils
from typing import Optional, Tuple

from .base_trainer import BaseTrainer

class PhysicsInformedTrainer(BaseTrainer):
    def __init__(self,
                 use_global_loss: bool = False,
                 global_mass_loss_scale: float = 1.0,
                 global_mass_loss_weight: float = 1.0,
                 use_local_loss: bool = False,
                 local_mass_loss_scale: float = 1.0,
                 local_mass_loss_weight: float = 1.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        ds: FloodEventDataset = self.dataloader.dataset
        self.use_physics_loss = use_global_loss or use_local_loss
        self.use_global_loss = use_global_loss
        self.use_local_loss = use_local_loss
        self.delta_t = ds.timestep_interval

        if self.use_global_loss:
            self.global_loss_func = GlobalMassConservationLoss(
                mode='train',
                normalizer=ds.normalizer,
                is_normalized=ds.is_normalized,
                delta_t=self.delta_t
            )
            self.global_loss_scaler = LossScaler(initial_scale=global_mass_loss_scale)
            self.global_mass_loss_weight = global_mass_loss_weight

        if self.use_local_loss:
            self.local_loss_func = LocalMassConservationLoss(
                mode='train',
                normalizer=ds.normalizer,
                is_normalized=ds.is_normalized,
                delta_t=self.delta_t,
            )
            self.local_loss_scaler = LossScaler(initial_scale=local_mass_loss_scale)
            self.local_mass_loss_weight = local_mass_loss_weight

    def _get_physics_loss(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        '''
        Compute and return the physics loss.

        Parameters:
            epoch (int): Current epoch number. Used for dynamic loss scaling.
            pred (Tensor): Predicted node values at current timestep.
            prev_node_pred (Tensor): Predicted node values at previous timestep.
            prev_edge_pred (Tensor): Predicted edge values at previous timestep.
            basis_loss (Tensor): The basis loss (e.g., prediction loss) to scale against.
            batch: The current data batch.
            current_timestep (Optional[int]): Current timestep index if training in autoregressive setting.
        '''
        if not self.use_global_loss and not self.use_local_loss:
            raise ValueError("At least one of global or local physics loss must be enabled.")

        global_physics_loss = torch.tensor(0.0)
        if self.use_global_loss:
            global_physics_loss = self._get_global_mass_loss(*args, **kwargs)

        local_physics_loss = torch.tensor(0.0)
        if self.use_local_loss:
            local_physics_loss = self._get_local_mass_loss(*args, **kwargs)

        return global_physics_loss, local_physics_loss

    def _get_global_mass_loss(self,
                              epoch: int,
                              pred: Tensor,
                              prev_node_pred: Tensor,
                              prev_edge_pred: Tensor,
                              basis_loss: Tensor,
                              batch,
                              current_timestep: Optional[int] = None) -> Tensor:
        total_rainfall = physics_utils.get_total_rainfall(batch, current_timestep)
        global_physics_loss = self.global_loss_func(pred, prev_node_pred, prev_edge_pred, total_rainfall, batch)
        global_physics_loss = self._scale_global_mass_loss(epoch, basis_loss, global_physics_loss)
        return global_physics_loss

    def _get_local_mass_loss(self,
                             epoch: int,
                             pred: Tensor,
                             prev_node_pred: Tensor,
                             prev_edge_pred: Tensor,
                             basis_loss: Tensor,
                             batch,
                             current_timestep: Optional[int] = None) -> Tensor:
        rainfall = physics_utils.get_rainfall(batch, current_timestep)
        local_physics_loss = self.local_loss_func(pred, prev_node_pred, prev_edge_pred, rainfall, batch)
        local_physics_loss = self._scale_local_mass_loss(epoch, basis_loss, local_physics_loss)
        return local_physics_loss

    def _log_epoch_physics_loss(self, global_mass_epoch_loss: float, local_mass_epoch_loss: float):
        if self.use_global_loss:
            self.training_stats.log(f'\tGlobal Physics Loss: {global_mass_epoch_loss:.4e}')
            self.training_stats.add_loss_component('global_physics_loss', global_mass_epoch_loss)

        if self.use_local_loss:
            self.training_stats.log(f'\tLocal Physics Loss: {local_mass_epoch_loss:.4e}')
            self.training_stats.add_loss_component('local_physics_loss', local_mass_epoch_loss)

# ========= Methods for scaling losses =========

    def _scale_global_mass_loss(self, epoch: int, basis_loss: Tensor, global_mass_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.global_loss_scaler.add_epoch_loss_ratio(basis_loss, global_mass_loss)
            scaled_global_physics_loss = self.global_loss_scaler.scale_loss(global_mass_loss)
        else:
            scaled_global_physics_loss = self.global_loss_scaler.scale_loss(global_mass_loss) * self.global_mass_loss_weight
        return scaled_global_physics_loss

    def _scale_local_mass_loss(self, epoch: int, basis_loss: Tensor, local_mass_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.local_loss_scaler.add_epoch_loss_ratio(basis_loss, local_mass_loss)
            scaled_local_physics_loss = self.local_loss_scaler.scale_loss(local_mass_loss)
        else:
            scaled_local_physics_loss = self.local_loss_scaler.scale_loss(local_mass_loss) * self.local_mass_loss_weight
        return scaled_local_physics_loss

    def _update_loss_scaler_for_epoch(self, epoch: int):
        if epoch < self.num_epochs_dyn_loss:
            if self.use_global_loss:
                self.global_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Global Mass Loss Weight to {self.global_loss_scaler.scale:.4e}')
            if self.use_local_loss:
                self.local_loss_scaler.update_scale_from_epoch()
                self.training_stats.log(f'\tAdjusted Local Mass Loss Weight to {self.local_loss_scaler.scale:.4e}')

    def _add_scaled_physics_loss_history(self):
        if self.use_global_loss:
            self.training_stats.add_additional_info('global_scaled_loss_ratios', self.global_loss_scaler.scaled_loss_ratio_history)

        if self.use_local_loss:
            self.training_stats.add_additional_info('local_scaled_loss_ratios', self.local_loss_scaler.scaled_loss_ratio_history)
