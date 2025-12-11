
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import GATConv, Sequential as PygSequential
from utils.model_utils import get_activation_func

from .base_edge_model import BaseEdgeModel

class EdgeGAT(BaseEdgeModel):
    '''
    GAT (Graph Attention Network)
    GNN utlizing attention mechanism for edges. Modified for Edge Prediction.
    '''
    def __init__(self,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 negative_slope: float = 0.2,
                 attn_bias: bool = True,
                 attn_residual: bool = True,
                 return_attn_weights: bool = False,
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.return_attn_weights = return_attn_weights

        conv_kwargs = {
            'heads': num_heads,
            'dropout': dropout,
            'add_self_loops': add_self_loops,
            'negative_slope': negative_slope,
            'bias': attn_bias,
            'residual': attn_residual,
            'return_attn_weights': self.return_attn_weights,
        }
        if self.use_edge_features:
            conv_kwargs['edge_dim'] = self.input_edge_features

        self.convs = self._make_gnn(input_size=self.input_size, output_size=self.output_size,
                              hidden_size=self.hidden_features, num_layers=self.num_layers,
                              activation=self.activation, use_edge_attr=self.use_edge_features,
                              device=self.device, **conv_kwargs)

    def _make_gnn(self, input_size: int, output_size: int, hidden_size: int = None,
                num_layers: int = 1, activation: str = None, use_edge_attr: bool = False,
                heads: int = 1, device: str = 'cpu', **conv_kwargs) -> Module:
        is_multihead = heads > 1

        if num_layers == 1:
            return GATLayer(input_size, output_size, activation, device, **conv_kwargs)

        layer_schema = 'x, edge_index -> x' if not use_edge_attr else 'x, edge_index, edge_attr -> x'
        input_schema = 'x, edge_index' if not use_edge_attr else 'x, edge_index, edge_attr'
        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (GATLayer(input_size, hidden_size, activation, use_edge_attr, device,
                      heads=heads, **conv_kwargs), layer_schema)
        ) # Input Layer

        for _ in range(num_layers-2):
            layers.append(
                (GATLayer((hidden_size * heads), hidden_size, activation, use_edge_attr, device,
                          heads=heads, **conv_kwargs), layer_schema)
            ) # Hidden Layers

        concat = not is_multihead
        layers.append(
            (GATLayer((hidden_size * heads), output_size, None, use_edge_attr, device,
                      heads=heads, concat=concat, **conv_kwargs), layer_schema)
        ) # Output Layer
        return PygSequential(input_schema, layers)

    def get_rollout_attn_weights(self):
        attn_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, GATLayer) and hasattr(module, 'get_rollout_attn_weights'):
                attn_weights[name] = module.get_rollout_attn_weights()
        return attn_weights


class GATLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = None,
                 use_edge_attr: bool = False,
                 device: str = 'cpu',
                 return_attn_weights: bool = False,
                 **conv_kwargs):
        super().__init__()
        self.conv = GATConv(in_channels=in_features, out_channels=out_features, **conv_kwargs).to(device)
        if activation is not None:
            self.activation = get_activation_func(activation, device=device)
        self.use_edge_attr = use_edge_attr
        self.return_attn_weights = return_attn_weights
        self.attn_weights = []

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        kwargs = {
            'x': x,
            'edge_index': edge_index,
        }
        if self.use_edge_attr:
            kwargs['edge_attr'] = edge_attr
        if self.return_attn_weights:
            kwargs['return_attention_weights'] = True

        out = self.conv(**kwargs)
        if self.return_attn_weights:
            x, attn_out = out
            attn_edge_index = attn_out[0].detach().cpu()
            attn_weights = attn_out[1].detach().cpu()
            self.attn_weights.append((attn_edge_index, attn_weights))
        else:
            x = out

        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

    def get_rollout_attn_weights(self):
        return self.attn_weights
