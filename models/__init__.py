from torch.nn import Module

from .edge_gnn_attn import EdgeGNNAttn
from .gat import GAT
from .gcn import GCN
from .node_edge_gnn import NodeEdgeGNN
from .node_edge_gnn_attn import NodeEdgeGNNAttn
from .node_gnn_attn import NodeGNNAttn

def model_factory(model_name: str, **kwargs) -> Module:
    if model_name == 'EdgeGNNAttn':
        return EdgeGNNAttn(**kwargs)
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    if model_name == 'NodeEdgeGNN':
        return NodeEdgeGNN(**kwargs)
    if model_name == 'NodeEdgeGNNAttn':
        return NodeEdgeGNNAttn(**kwargs)
    if model_name == 'NodeGNNAttn':
        return NodeGNNAttn(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = ['GAT', 'GCN', 'model_factory']
