from torch.nn import Module

from .gat import GAT
from .gcn import GCN

def model_factory(model_name: str, **kwargs) -> Module:
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = ['GAT', 'GCN', 'model_factory']
