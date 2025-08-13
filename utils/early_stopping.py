from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union

class EarlyStopping:
    def __init__(self,
                 patience: int = 7,
                 min_delta: int = 0,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: Union[Tensor, Tuple[Tensor]], model: Module) -> bool:
        if self.best_loss is None: # Init call
            self.best_loss = val_loss
            self.save_checkpoint(model)
            return False

        if isinstance(val_loss, tuple):
            is_improving = any(val_loss[i] < self.best_loss[i] - self.min_delta for i in range(len(self.best_loss)))
        else: # Assumed to be a single Tensor
            is_improving = val_loss < self.best_loss - self.min_delta

        if is_improving:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model: Module) -> None:
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
