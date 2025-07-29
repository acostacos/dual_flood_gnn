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
from typing import List, Optional

class NodeEdgeGNNGAT(BaseModel):
    def __init__(self,
                 input_features: int = None,
                 input_edge_features: int = None,
                 output_features: int = None,
                 output_edge_features: int = None,
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
        if output_edge_features is None:
            output_edge_features = self.output_edge_features

        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features
        input_edge_size = hidden_features if self.with_encoder else input_edge_features
        output_edge_size = hidden_features if self.with_decoder else output_edge_features

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
                              output_edge_size=output_edge_size, hidden_size=hidden_features, num_layers=num_layers,
                              activation=activation, **conv_kwargs)
        self.convs = self.convs.to(self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=output_edge_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, input_size: int, output_size: int, input_edge_size: int, output_edge_size: int,
                  hidden_size: int = None, num_layers: int = 1, activation: str = None,
                  heads: int = 1, **conv_kwargs) -> Module:
        is_multihead = heads > 1

        if num_layers == 1:
            return GATLayer(input_size, output_size, input_edge_size, output_edge_size, activation, **conv_kwargs)

        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (GATLayer(input_size, hidden_size, input_edge_size, output_edge_size, activation, heads=heads, **conv_kwargs),
             'x, edge_index, edge_attr -> x, edge_attr')
        ) # Input Layer

        for _ in range(num_layers-2):
            layers.append(
                (GATLayer((hidden_size * heads), hidden_size, input_edge_size, output_edge_size, activation, heads=heads, **conv_kwargs),
                 'x, edge_index, edge_attr -> x, edge_attr')
            ) # Hidden Layers

        concat = not is_multihead
        layers.append(
            (GATLayer((hidden_size * heads), hidden_size, input_edge_size, output_edge_size, activation, heads=heads, concat=concat,
                      **conv_kwargs), 'x, edge_index, edge_attr -> x, edge_attr')
        ) # Output Layer
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        x0 = x

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x, edge_attr

class GATLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 in_edge_features: int,
                 out_edge_features: int,
                 activation: str = None,
                 return_attn_weights: bool = False,
                 **conv_kwargs):
        super().__init__()
        self.conv = GATConv(in_features, out_features, in_edge_features, out_edge_features, **conv_kwargs)
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

        out, edge_attr = self.conv(**kwargs)
        if self.return_attn_weights:
            x, attn_out = out
            attn_edge_index = attn_out[0].detach().cpu()
            attn_weights = attn_out[1].detach().cpu()
            self.attn_weights.append((attn_edge_index, attn_weights))
        else:
            x = out

        if hasattr(self, 'activation'):
            x = self.activation(x)
            edge_attr = self.activation(edge_attr)
        return x, edge_attr

class GATConv(MessagePassing):
    '''Based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py'''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_in_features: int = None,
                 edge_out_features: int = None,
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
        self.inspector.inspect_signature(self.attention)
        self._attn_user_args: List[str] = self.inspector.get_param_names(
            'attention', exclude=self.special_args)

        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.has_edge_features = edge_in_features is not None
        self.out_features = out_features

        self.lin = Linear(in_features, out_features, bias=False, device=device) 
        self.node_attn = Linear((out_features * 3), 1, bias=False, device=device)

        self.msg_lin = Linear((out_features * 2), edge_out_features, bias=False, device=device)

        node_update_input_size = (out_features * 2)
        self.node_update_mlp = Linear(node_update_input_size, out_features, bias=False, device=device)

        self.out_lin = Linear(out_features, out_features, bias=False, device=device)
        self.edge_attn = Linear((out_features * 2), 1, bias=False, device=device)

        edge_update_input_size = (out_features * 2)
        self.edge_update_mlp = Linear(edge_update_input_size, edge_out_features, bias=False, device=device)

        if self.has_edge_features:
            self.lin_edge = Linear(edge_in_features, out_features, bias=False, device=device)

        total_out_channels = out_features
        if residual:
            self.residual = Linear(in_features, total_out_channels, bias=False, device=device)
            self.residual_edge = Linear(edge_in_features, total_out_channels, bias=False, device=device)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
            self.bias_edge = Parameter(torch.empty(total_out_channels))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None):
        res: Optional[Tensor] = None
        res_edge: Optional[Tensor] = None
        if hasattr(self, 'residual'):
            res = self.residual(x)
            res_edge = self.residual_edge(edge_attr)

        x = self.lin(x)
        edge_attr = self.lin_edge(edge_attr)

        if self.add_self_loops:
            # Only add self-loops for nodes that are both source and target
            num_nodes = x.shape[0]
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes)

        alpha = self.compute_attention(edge_index, x=x, edge_attr=edge_attr)

        out, edge_out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr)

        if self.add_self_loops:
            edge_index, edge_out = remove_self_loops(edge_index, edge_out)

        if res is not None:
            out = out + res
            edge_out = edge_out + res_edge

        if hasattr(self, 'bias'):
            out = out + self.bias
            edge_out = edge_out + self.bias_edge

        # TODO: Support returning attention weights
        # if isinstance(return_attention_weights, bool):
        #     if is_torch_sparse_tensor(edge_index):
        #         # TODO TorchScript requires to return a tuple
        #         adj = set_sparse_value(edge_index, alpha)
        #         return out, (adj, alpha)
        #     else:
        #         return out, (edge_index, alpha)

        return out, edge_out

    def compute_attention(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._attn_user_args, edge_index, mutable_size, kwargs)
        edge_kwargs = self.inspector.collect_param_data('attention', coll_dict)
        alpha = self.attention(**edge_kwargs)
        return alpha

    def attention(self,
                  x_i: Tensor,
                  x_j: Tensor,
                  edge_attr:
                  Tensor,
                  index: Tensor,
                  ptr,
                  dim_size: Optional[int]) -> Tensor:
        alpha = self.node_attn(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggr = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        node_out = self.update(aggr, **update_kwargs)

        out = self.out_lin(node_out)
        edge_kwargs = {**kwargs, 'out': out, 'msg': msg}
        edge_coll_dict = self._collect(self._edge_user_args, edge_index, mutable_size, edge_kwargs)
        edge_update_kwargs = self.inspector.collect_param_data('edge_update', edge_coll_dict)
        edge_out = self.edge_update(**edge_update_kwargs)

        return node_out, edge_out

    def message(self, x_j: Tensor, edge_attr: Tensor, alpha: Tensor) -> Tensor:
        return alpha * torch.cat([x_j, edge_attr], dim=-1)

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.node_update_mlp(aggr_out)

    def edge_update(self, msg: Tensor, out_i: Tensor, out_j: Tensor, index, ptr, dim_size) -> Tensor:
        beta = self.edge_attn(torch.cat([out_i, out_j], dim=-1))
        beta = F.leaky_relu(beta, self.negative_slope)
        beta = softmax(beta, index, ptr, dim_size)
        beta = F.dropout(beta, p=self.dropout, training=self.training)
        return self.edge_update_mlp(beta * msg)
