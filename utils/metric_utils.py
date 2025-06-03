import torch

from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor) -> Tensor:
    return l1_loss(pred, target)

def NSE(pred: Tensor, target: Tensor) -> Tensor:
    '''Nash Sutcliffe Efficiency'''
    model_sse = torch.sum((target - pred)**2)
    mean_model_sse = torch.sum((target - target.mean())**2)
    return 1 - (model_sse / mean_model_sse)

def CSI(binary_pred: Tensor, binary_target: Tensor):
    TP = (binary_pred & binary_target).sum() #true positive
    # TN = (~binary_pred & ~binary_target).sum() #true negative
    FP = (binary_pred & ~binary_target).sum() #false positive
    FN = (~binary_pred & binary_target).sum() #false negative

    return TP / (TP + FN + FP)
