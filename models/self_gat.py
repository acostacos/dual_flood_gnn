import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Identity, Linear, Parameter, Module
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from torch_geometric.utils import add_self_loops, is_torch_sparse_tensor, remove_self_loops, softmax
from torch_geometric.utils.sparse import set_sparse_value
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel
from typing import Optional

class SelfGAT(BaseModel):
    def __init__(self,
                 input_features: int = None,
                 input_edge_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,

                 # Attention Parameters
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 negative_slope: float = 0.2,
                 attn_bias: bool = True,
                 attn_residual: bool = True,
                 return_attn_weights: bool = False,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        if input_edge_features is None:
            input_edge_features = self.input_edge_features

        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features
        input_edge_size = hidden_features if self.with_encoder else input_edge_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        conv_kwargs = {
            'heads': num_heads,
            'dropout': dropout,
            'add_self_loops': add_self_loops,
            'negative_slope': negative_slope,
            'bias': attn_bias,
            'residual': attn_residual,
            'return_attn_weights': return_attn_weights,
        }

        self.convs = self._make_gnn(input_size=input_size, output_size=output_size, input_edge_size=input_edge_size,
                              hidden_size=hidden_features, num_layers=num_layers,
                              activation=activation, **conv_kwargs)
        self.convs = self.convs.to(self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, input_size: int, output_size: int, input_edge_size: int,
                  hidden_size: int = None, num_layers: int = 1, activation: str = None,
                  heads: int = 1, **conv_kwargs) -> Module:
        is_multihead = heads > 1

        if num_layers == 1:
            return GATLayer(input_size, output_size, input_edge_size, activation, **conv_kwargs)

        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (GATLayer(input_size, hidden_size, input_edge_size, activation, heads=heads, **conv_kwargs),
             'x, edge_index, edge_attr -> x')
        ) # Input Layer

        for _ in range(num_layers-2):
            layers.append(
                (GATLayer((hidden_size * heads), hidden_size, input_edge_size, activation, heads=heads, **conv_kwargs),
                 'x, edge_index, edge_attr -> x')
            ) # Hidden Layers

        concat = not is_multihead
        layers.append(
            (GATLayer((hidden_size * heads), hidden_size, input_edge_size, activation, heads=heads, concat=concat,
                      **conv_kwargs), 'x, edge_index, edge_attr -> x')
        ) # Output Layer
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        x0 = x

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x

class GATLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 in_edge_features: int,
                 activation: str = None,
                 return_attn_weights: bool = False,
                 **conv_kwargs):
        super().__init__()
        self.conv = GATConv(in_features, out_features, in_edge_features, **conv_kwargs)
        if activation is not None:
            self.activation = get_activation_func(activation)
        self.return_attn_weights = return_attn_weights
        self.attn_weights = []

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        kwargs = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        }

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

class GATConv(MessagePassing):
    '''Based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py'''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_in_features: int = None,
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 negative_slope: float = 0.2,
                 bias: bool = True,
                 residual: bool = True,
                 return_attn_weights: bool = False,
                 device: str = 'cpu'):
        super().__init__(aggr='sum', node_dim=0)
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.has_edge_features = edge_in_features is not None
        self.out_features = out_features

        self.lin = Linear(in_features, (heads*out_features), bias=False, device=device)
        self.attn_s = Linear(out_features, 1)
        self.attn_t = Linear(out_features, 1)

        if self.has_edge_features:
            self.lin_edge = Linear(edge_in_features, (heads*out_features), bias=False, device=device)
            self.attn_edge = Linear(out_features, 1)

        total_out_channels = out_features * (heads if concat else 1)
        if residual:
            self.residual = Linear(in_features, total_out_channels, bias=False, device=device)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None):
        res: Optional[Tensor] = None
        if hasattr(self, 'residual'):
            res = self.residual(x)

        x_s = x_t = self.lin(x).view(-1, self.heads, self.out_features)
        x = (x_s, x_t)

        alpha_s = self.attn_s(x_s).sum(dim=-1)
        alpha_t = self.attn_t(x_t).sum(dim=-1)
        alpha = (alpha_s, alpha_t)

        if self.add_self_loops:
            # Only add self-loops for nodes that are both source and target
            num_nodes = x_s.shape[0]
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if hasattr(self, 'bias'):
            out = out + self.bias

        # TODO: Support returning attention weights
        # if isinstance(return_attention_weights, bool):
        #     if is_torch_sparse_tensor(edge_index):
        #         # TODO TorchScript requires to return a tuple
        #         adj = set_sparse_value(edge_index, alpha)
        #         return out, (adj, alpha)
        #     else:
        #         return out, (edge_index, alpha)

        return out

    def edge_update(self,
                    alpha_j: Tensor,
                    alpha_i: Tensor,
                    edge_attr: Tensor,
                    index: Tensor,
                    ptr,
                    dim_size: Optional[int]) -> Tensor:
        # Sum is used to emulate concatenation
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha

        if self.has_edge_features and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_features)
            alpha_edge = self.attn_edge(edge_attr).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j
