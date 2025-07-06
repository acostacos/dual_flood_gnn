from torch.nn import Module

from .gat import GAT
from .gcn import GCN
from .node_edge_gnn import NodeEdgeGNN

def model_factory(model_name: str, **kwargs) -> Module:
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    if model_name == 'NodeEdgeGNN':
        return NodeEdgeGNN(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = ['GAT', 'GCN', 'model_factory']
