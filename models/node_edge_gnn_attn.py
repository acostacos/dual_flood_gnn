import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Identity, Linear, Parameter, Module
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import softmax
from typing import List, Optional, Tuple
from utils.model_utils import make_mlp

from .base_model import BaseModel

class NodeEdgeGNNAttn(BaseModel):
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
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 attn_mlp_layers: float = 2,
                 attn_bias: bool = True,

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

        # Processor
        conv_kwargs = {
            'input_size': input_size,
            'output_size': output_size,
            'input_edge_size': input_edge_size,
            'output_edge_size': output_edge_size,
            'hidden_size': hidden_features,
            'dropout': dropout,
            'negative_slope': negative_slope,
            'mlp_layers': attn_mlp_layers,
            'bias': attn_bias,
            'gnn_layers': num_layers,
            'activation': activation,
        }
        self.convs = self._make_gnn(**conv_kwargs)
        self.convs = self.convs.to(self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=output_edge_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, input_size: int, output_size: int, input_edge_size: int, output_edge_size: int,
                  hidden_size: int = None, gnn_layers: int = 1, **conv_kwargs) -> Module:
        if gnn_layers == 1:
            return NodeEdgeAttnConv(input_size, output_size, input_edge_size, output_edge_size, **conv_kwargs)

        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (NodeEdgeAttnConv(input_size, hidden_size, input_edge_size, hidden_size, **conv_kwargs),
             'x, edge_index, edge_attr -> x, edge_attr')
        ) # Input Layer

        for _ in range(gnn_layers-2):
            layers.append(
                (NodeEdgeAttnConv(hidden_size, hidden_size, hidden_size, hidden_size, **conv_kwargs),
                 'x, edge_index, edge_attr -> x, edge_attr')
            ) # Hidden Layers

        layers.append(
            (NodeEdgeAttnConv(hidden_size, output_size, hidden_size, output_edge_size, **conv_kwargs),
             'x, edge_index, edge_attr -> x, edge_attr')
        ) # Output Layer
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x0, edge_attr0 = x.clone(), edge_attr.clone()

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])
            edge_attr = edge_attr + self.residual(edge_attr0[:, -self.output_edge_features:])

        return x, edge_attr

class NodeEdgeAttnConv(MessagePassing):
    '''Based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py'''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_in_features: int,
                 edge_out_features: int,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 mlp_layers: float = 2,
                 bias: bool = True,
                 activation: str = None):
        super().__init__(aggr='sum', node_dim=0)
        self.inspector.inspect_signature(self.attention)
        self._attn_user_args: List[str] = self.inspector.get_param_names(
            'attention', exclude=self.special_args)

        self.dropout = dropout
        self.negative_slope = negative_slope
        self.has_bias = bias

        self.lin = Linear(in_features, out_features, bias=False) 
        self.edge_lin = Linear(edge_in_features, edge_out_features, bias=False)
        self.node_attn = Linear((out_features * 2 + edge_out_features), 1, bias=False)
        self.edge_attn = Linear((out_features * 2), 1, bias=False)

        msg_input_size = (out_features * 2 + edge_out_features)
        msg_hidden_size = msg_input_size * 2
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out_features,
                            hidden_size=msg_hidden_size, num_layers=mlp_layers,
                            activation=activation)

        node_update_input_size = (edge_out_features)
        node_update_hidden_size = node_update_input_size * 2
        self.node_update_mlp = make_mlp(input_size=node_update_input_size, output_size=out_features,
                                 hidden_size=node_update_hidden_size, num_layers=mlp_layers,
                                 activation=activation)

        edge_update_input_size = (edge_out_features)
        edge_update_hidden_size = edge_update_input_size * 2
        self.edge_update_mlp = make_mlp(input_size=edge_update_input_size, output_size=edge_out_features,
                                 hidden_size=edge_update_hidden_size, num_layers=mlp_layers,
                                 activation=activation)

        if self.has_bias:
            self.bias = Parameter(torch.empty(out_features))
            self.bias_edge = Parameter(torch.empty(edge_out_features))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.lin.reset_parameters()
        self.edge_lin.reset_parameters()
        self.node_attn.reset_parameters()
        self.edge_attn.reset_parameters()

        for layer in self.msg_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.node_update_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.edge_update_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if self.has_bias:
            zeros(self.bias)
            zeros(self.bias_edge)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x = self.lin(x)
        edge_attr = self.edge_lin(edge_attr)

        alpha = self.compute_attention(edge_index, x=x, edge_attr=edge_attr)

        out, msg = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr)

        edge_out = self.edge_updater(edge_index, out=out, msg=msg)

        if self.has_bias:
            out = out + self.bias
            edge_out = edge_out + self.bias_edge

        return out, edge_out

    def compute_attention(self, edge_index: Tensor, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._attn_user_args, edge_index, mutable_size, kwargs)
        edge_kwargs = self.inspector.collect_param_data('attention', coll_dict)
        alpha = self.attention(**edge_kwargs)
        return alpha

    def attention(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr, dim_size: Optional[int]) -> Tensor:
        alpha = self.node_attn(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def propagate(self, edge_index: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        node_msg = kwargs['alpha'] * msg
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggr = self.aggregate(node_msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        node_out = self.update(aggr, **update_kwargs)

        return node_out, msg

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.msg_mlp(torch.cat([x_i, edge_attr, x_j], dim=-1))

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.node_update_mlp(aggr_out)

    def edge_update(self, msg: Tensor, out_i: Tensor, out_j: Tensor, index: Tensor, ptr, dim_size: Optional[int]) -> Tensor:
        beta = self.edge_attn(torch.cat([out_i, out_j], dim=-1))
        beta = F.leaky_relu(beta, self.negative_slope)
        beta = softmax(beta, index, ptr, dim_size)
        beta = F.dropout(beta, p=self.dropout, training=self.training)
        return self.edge_update_mlp(beta * msg)
