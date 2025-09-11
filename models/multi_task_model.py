import torch

from torch.nn import Parameter, ParameterList

from .base_model import BaseModel

class MultiTaskModel(BaseModel):
    def __init__(self,
                 num_tasks: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.multi_task_loss_params = ParameterList(
            [Parameter(torch.rand(1, device=self.device) * 0.1) for _ in range(num_tasks)]
        )

    def get_multi_task_loss_params(self) -> ParameterList:
        return self.multi_task_loss_params
