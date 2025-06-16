import os
import time
import numpy as np
from . import Logger

class TrainingStats:
    def __init__(self, logger: Logger = None):
        self.total_epoch_loss = []
        self.epoch_loss_components = {}
        self.additional_info = {}

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

    def start_train(self):
        self.train_start_time = time.time()

    def end_train(self):
        self.train_end_time = time.time()

    def get_train_time(self):
        return self.train_end_time - self.train_start_time

    def add_loss(self, loss):
        self.total_epoch_loss.append(loss)

    def add_loss_component(self, key: str, loss):
        if key not in self.epoch_loss_components:
            self.epoch_loss_components[key] = []
        self.epoch_loss_components[key].append(loss)

    def add_additional_info(self, key: str, info):
        self.additional_info[key] = info

    def print_stats_summary(self):
        if len(self.total_epoch_loss) > 0:
            self.log(f'Final training Loss: {self.total_epoch_loss[-1]:.4e}')
            np_epoch_loss = np.array(self.total_epoch_loss)
            self.log(f'Average training Loss: {np_epoch_loss.mean():.4e}')
            self.log(f'Minimum training Loss: {np_epoch_loss.min():.4e}')
            self.log(f'Maximum training Loss: {np_epoch_loss.max():.4e}')

        if self.train_start_time is not None and self.train_end_time is not None:
            self.log(f'Total training time: {self.get_train_time():.4f} seconds')

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        stats = {
            'train_epoch_loss': np.array(self.total_epoch_loss),
            'train_time': self.get_train_time(),
        }
        np_loss_components = {k: np.array(v) for k, v in self.epoch_loss_components.items()}
        stats.update(np_loss_components)
        stats.update(self.additional_info)
        np.savez(filepath, **stats)
        self.log(f'Saved training stats to: {filepath}')
