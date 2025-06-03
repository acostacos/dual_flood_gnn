import os
import time
import matplotlib.pyplot as plt
import numpy as np
from . import Logger

class TrainingStats:
    def __init__(self, logger: Logger = None):
        self.train_epoch_loss = []

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
        self.train_epoch_loss.append(loss)

    def print_stats_summary(self):
        if len(self.train_epoch_loss) > 0:
            self.log(f'Final training Loss: {self.train_epoch_loss[-1]:.4f}')
            np_epoch_loss = np.array(self.train_epoch_loss)
            self.log(f'Average training Loss: {np_epoch_loss.mean():.4f}')
            self.log(f'Minimum training Loss: {np_epoch_loss.min():.4f}')
            self.log(f'Maximum training Loss: {np_epoch_loss.max():.4f}')

        if self.train_start_time is not None and self.train_end_time is not None:
            self.log(f'Total training time: {self.get_train_time():.4f} seconds')

    def plot_train_loss(self):
        plt.plot(self.train_epoch_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        stats = {
            'train_epoch_loss': np.array(self.train_epoch_loss),
        }
        np.savez(filepath, **stats)
        self.log(f'Saved training stats to: {filepath}')
