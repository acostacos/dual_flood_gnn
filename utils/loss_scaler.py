import numpy as np

from torch import Tensor

class LossScaler:
    def __init__(self, initial_scale: float = 1.0):
        self.scale = initial_scale
        self.history = [initial_scale]
        self.epoch_loss_ratios = []
    
    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale
    
    def get_ratio(self, basis_loss: Tensor, loss: Tensor) -> float:
        basis_loss, loss = basis_loss.item(), loss.item()
        return basis_loss / (loss + 1e-8)

    def add_epoch_loss_ratio(self, basis_loss: Tensor, loss: Tensor):
        ratio = self.get_ratio(basis_loss, loss)
        self.epoch_loss_ratios.append(ratio)

    def update_scale_from_epoch(self):
        self.scale = np.sum(self.epoch_loss_ratios) / len(self.epoch_loss_ratios)
        self.history.append(self.scale)
        self.epoch_loss_ratios = []
