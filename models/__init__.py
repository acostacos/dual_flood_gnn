from torch.nn import Module

from .edge_gat import EdgeGAT
from .edge_gcn import EdgeGCN
from .edge_gnn import EdgeGNN
from .gat import GAT
from .gcn import GCN
from .dual_flood_gnn import DUALFloodGNN
from .node_edge_gnn_transformer import NodeEdgeGNNTransformer
from .node_edge_gnn_attn import NodeEdgeGNNAttn
from .node_gnn import NodeGNN

def model_factory(model_name: str, *args, **kwargs) -> Module:
    if model_name == 'DUALFloodGNN':
        return DUALFloodGNN(*args, **kwargs)
    if model_name == 'EdgeGAT':
        return EdgeGAT(*args, **kwargs)
    if model_name == 'EdgeGCN':
        return EdgeGCN(*args, **kwargs)
    if model_name == 'EdgeGNN':
        return EdgeGNN(*args, **kwargs)
    if model_name == 'GCN':
        return GCN(*args, **kwargs)
    if model_name == 'GAT':
        return GAT(*args, **kwargs)
    if model_name == 'NodeEdgeGNNAttn':
        return NodeEdgeGNNAttn(*args, **kwargs)
    if model_name == 'NodeEdgeGNNTransformer':
        return NodeEdgeGNNTransformer(*args, **kwargs)
    if model_name == 'NodeGNN':
        return NodeGNN(*args, **kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = [
    'DUALFloodGNN',
    'EdgeGAT',
    'EdgeGCN',
    'EdgeGNN',
    'GAT',
    'GCN',
    'NodeEdgeGNNAttn',
    'NodeEdgeGNNTransformer',
    'NodeGNN',
    'model_factory',
]
