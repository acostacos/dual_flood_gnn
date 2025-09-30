import numpy as np

from torch import Tensor
from typing import Optional

class LossScaler:
    def __init__(self, initial_scale: Optional[float] = None):
        self.scale = initial_scale or 1.0  # Default to 1.0 if not provided

        # Dynamic loss scaling
        self.absolute_scale_history = [initial_scale]
        self.scaled_loss_ratio_history = []
        self.epoch_loss_ratios = []
        self.epoch_scaled_loss_ratios = []

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def add_epoch_loss_ratio(self, basis_loss: float, loss: float):
        ratio = basis_loss / (loss + 1e-8)
        self.epoch_loss_ratios.append(ratio)

        scaled_loss = self.scale_loss(loss)
        scaled_loss_ratio = basis_loss / (scaled_loss + 1e-8)
        self.epoch_scaled_loss_ratios.append(scaled_loss_ratio)

    def update_scale_from_epoch(self):
        scaled_loss_ratio = np.mean(self.epoch_scaled_loss_ratios)
        self.scaled_loss_ratio_history.append(scaled_loss_ratio)

        self.scale = np.mean(self.epoch_loss_ratios)
        self.absolute_scale_history.append(self.scale)

        self.epoch_loss_ratios = []
        self.epoch_scaled_loss_ratios = []
