from torch.nn import Module

from .base_model import BaseModel
from .base_node_model import BaseNodeModel
from .base_edge_model import BaseEdgeModel
from .edge_gnn import EdgeGNN
from .gat import GAT, EdgeGAT
from .gcn import GCN, EdgeGCN
from .gin import GIN, EdgeGIN
from .gine import GINE, EdgeGINE
from .graphsage import GraphSAGE, EdgeGraphSAGE
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
    if model_name == 'EdgeGIN':
        return EdgeGIN(*args, **kwargs)
    if model_name == 'EdgeGINE':
        return EdgeGINE(*args, **kwargs)
    if model_name == 'EdgeGraphSAGE':
        return EdgeGraphSAGE(*args, **kwargs)
    if model_name == 'EdgeGNN':
        return EdgeGNN(*args, **kwargs)
    if model_name == 'GCN':
        return GCN(*args, **kwargs)
    if model_name == 'GAT':
        return GAT(*args, **kwargs)
    if model_name == 'GIN':
        return GIN(*args, **kwargs)
    if model_name == 'GINE':
        return GINE(*args, **kwargs)
    if model_name == 'GraphSAGE':
        return GraphSAGE(*args, **kwargs)
    if model_name == 'NodeEdgeGNNAttn':
        return NodeEdgeGNNAttn(*args, **kwargs)
    if model_name == 'NodeEdgeGNNTransformer':
        return NodeEdgeGNNTransformer(*args, **kwargs)
    if model_name == 'NodeGNN':
        return NodeGNN(*args, **kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

__all__ = [
    'BaseModel',
    'BaseNodeModel',
    'BaseEdgeModel',
    'DUALFloodGNN',
    'EdgeGAT',
    'EdgeGCN',
    'EdgeGIN',
    'EdgeGINE',
    'EdgeGraphSAGE',
    'EdgeGNN',
    'GAT',
    'GCN',
    'GIN',
    'GINE',
    'GraphSAGE',
    'NodeEdgeGNNAttn',
    'NodeEdgeGNNTransformer',
    'NodeGNN',
    'model_factory',
]
