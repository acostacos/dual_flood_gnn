from torch.nn import Module

from .edge_gnn_attn import EdgeGNNAttn
from .gat import GAT
from .gcn import GCN
from .hydrographnet import HydroGraphNet
from .node_edge_gnn import NodeEdgeGNN
from .node_edge_gnn_attn import NodeEdgeGNNAttn
from .node_gnn_attn import NodeGNNAttn

def model_factory(model_name: str, *args, **kwargs) -> Module:
    if model_name == 'EdgeGNNAttn':
        return EdgeGNNAttn(*args, **kwargs)
    if model_name == 'GCN':
        return GCN(*args, **kwargs)
    if model_name == 'GAT':
        return GAT(*args, **kwargs)
    if model_name == 'HydroGraphNet':
        return HydroGraphNet(*args, **kwargs)
    if model_name == 'NodeEdgeGNN':
        return NodeEdgeGNN(*args, **kwargs)
    if model_name == 'NodeEdgeGNNAttn':
        return NodeEdgeGNNAttn(*args, **kwargs)
    if model_name == 'NodeGNNAttn':
        return NodeGNNAttn(*args, **kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = [
    'EdgeGNNAttn',
    'GAT',
    'GCN',
    'HydroGraphNet',
    'NodeEdgeGNN',
    'NodeEdgeGNNAttn',
    'NodeGNNAttn',
    'model_factory',
]
