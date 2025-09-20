import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from torch_geometric.utils import softmax
from typing import Optional, Tuple
from utils.model_utils import make_mlp

from .base_model import BaseModel

class NodeEdgeGNNAttn(BaseModel):
    '''
    NodeEdgeGNN with Local Attention. Doesn't seem to work well with node and edge prediction at the same time. May need further testing and tuning.
    '''
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
                                            activation=encoder_activation, device=self.device, bias=False)
            self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device, bias=False)

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
            'gnn_layers': num_layers,
            'activation': activation,
        }
        self.convs = self._make_gnn(**conv_kwargs)
        self.convs = self.convs.to(self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, device=self.device, bias=False)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=output_edge_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, device=self.device, bias=False)

    def _make_gnn(self, input_size: int, output_size: int, input_edge_size: int, output_edge_size: int,
                  hidden_size: int = None, gnn_layers: int = 1, **conv_kwargs) -> Module:
        if gnn_layers == 1:
            return NodeEdgeAttnConv(input_size, output_size, input_edge_size, output_edge_size, **conv_kwargs)

        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (NodeEdgeAttnConv(input_size, hidden_size, input_edge_size, hidden_size, hidden_size, **conv_kwargs),
             'x, edge_index, edge_attr -> x, edge_attr')
        ) # Input Layer

        for _ in range(gnn_layers-2):
            layers.append(
                (NodeEdgeAttnConv(hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, **conv_kwargs),
                 'x, edge_index, edge_attr -> x, edge_attr')
            ) # Hidden Layers

        layers.append(
            (NodeEdgeAttnConv(hidden_size, output_size, hidden_size, output_edge_size, hidden_size, **conv_kwargs),
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

        return x, edge_attr

class NodeEdgeAttnConv(MessagePassing):
    '''
    Based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py

    Message =  attn * MLP(node_i, edge_attr, node_j)
    Aggregate = sum
    Update = MLP(aggregated_message)
    '''
    def __init__(self, node_in_channels: int, node_out_channels: int,
                 edge_in_channels: int, edge_out_channels: int, hidden_size: int,
                 dropout: float = 0.0, negative_slope: float = 0.2,
                 mlp_layers: int = 2, activation: str = 'relu',
                 residual: bool = True, bias: bool = False, device: str = 'cpu'):
        super().__init__(aggr='sum')
        self.residual = residual
        self.negative_slope = 0.2
        self.dropout = 0.0

        self.node_lin = Linear(node_in_channels, hidden_size, bias=False) 
        self.edge_lin = Linear(edge_in_channels, hidden_size, bias=False)
        self.attn = Linear((node_in_channels * 2 + edge_out_channels), 1, bias=False)

        self.node_mlp = make_mlp(input_size=node_in_channels, output_size=node_out_channels, hidden_size=hidden_size,
                                 num_layers=mlp_layers, activation=activation, bias=bias, device=device)

        input_size = (node_in_channels * 2 + edge_in_channels)
        self.edge_mlp = make_mlp(input_size=input_size, output_size=edge_out_channels, hidden_size=hidden_size,
                                 num_layers=mlp_layers, activation=activation, bias=bias, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.node_lin(x)
        edge_attr = self.edge_lin(edge_attr)

        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, alpha=alpha)

    def edge_update(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr, dim_size: Optional[int]) -> Tensor:
        alpha = self.attn(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        node_msg = msg * kwargs['alpha']
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggr = self.aggregate(node_msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(aggr, **update_kwargs)

        return out, msg

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        cat_feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        msg = self.edge_mlp(cat_feats)
        if self.residual:
            msg = msg + edge_attr
        return msg

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        out = self.node_mlp(aggr)
        if self.residual:
            out = out + x
        return out
