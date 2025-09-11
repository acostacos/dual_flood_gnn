import torch

from torch import Tensor
from torch.nn import ParameterList, Module
from typing import List

class MultiTaskLossBalancer(Module):
    '''
    Based on the paper "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" by Kendall et al. (2018)
    Implementation based from https://github.com/oscarkey/multitask-learning/
    '''
    def __init__(self, num_tasks: int, loss_uncertainties: ParameterList = None):
        super().__init__()

        assert num_tasks == len(loss_uncertainties), "Number of tasks must match the number of loss uncertainties provided."
        self.log_vars = loss_uncertainties

    def forward(self, *losses: List[Tensor]) -> Tensor:
        total_loss = 0
        for i, task_loss in enumerate(losses):
            log_var = self.log_vars[i]
            weighted_loss = 0.5 * (torch.exp(-log_var) * task_loss + log_var)
            total_loss += weighted_loss
        return total_loss

    def get_task_weight(self, task_index: int) -> float:
        num_tasks = len(self.log_vars)
        if task_index < 0 or task_index >= num_tasks:
            raise ValueError(f"Task index {task_index} is out of bounds for number of tasks {num_tasks}.")
        return self.log_vars[task_index].item()

