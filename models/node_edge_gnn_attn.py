import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Identity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from torch_geometric.utils import softmax
from typing import List
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class NodeEdgeGNNAttn(BaseModel):
    '''
    Model that uses message passing to update both node and edge features. Can predict for both simultaneously.
    '''
    def __init__(self,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,
                 mlp_layers: int = 2,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_residual = residual


        # Encoder
        encoder_decoder_hidden = hidden_features*2
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=encoder_decoder_hidden, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        input_node_size = hidden_features if self.with_encoder else self.input_node_features
        output_node_size = hidden_features if self.with_decoder else self.output_node_features
        input_edge_size = hidden_features if self.with_encoder else self.input_edge_features
        output_edge_size = hidden_features if self.with_decoder else self.output_edge_features
        self.convs = self._make_gnn(input_node_size=input_node_size, output_node_size=output_node_size,
                                    input_edge_size=input_edge_size, output_edge_size=output_edge_size,
                                    num_layers=num_layers, mlp_layers=mlp_layers, activation=activation, device=self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=self.output_edge_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if self.with_residual:
            self.residual = Identity()
            self.res_activation = get_activation_func(activation, device=self.device)

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  num_layers: int, mlp_layers: int, activation: str, device: str):
        if num_layers == 1:
            return NodeEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                                num_layers=mlp_layers, activation=activation, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append((
                NodeEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr',
            ))
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        x0, edge_attr0 = x, edge_attr

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        if self.with_residual:
            x = self.res_activation(x + self.residual(x0[:, -self.output_node_features:]))
            edge_attr = self.res_activation(edge_attr + self.residual(edge_attr0[:, -self.output_edge_features:]))

        return x, edge_attr

class NodeEdgeConv(MessagePassing):
    """
    Message = MLP
    Aggregate = sum
    Update = MLP
    """
    def __init__(self, node_in_channels: int, node_out_channels: int, edge_in_channels: int, edge_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        self.inspector.inspect_signature(self.attention)
        self._attn_user_args: List[str] = self.inspector.get_param_names(
            'attention', exclude=self.special_args)

        hidden_size = 32
        # self.node_mlp = Linear(node_in_channels, hidden_size, bias=False, device=device)
        # self.edge_mlp = Linear(edge_in_channels, hidden_size, bias=False, device=device)

        attn_input_size = (hidden_size * 3)
        self.node_attn = Linear(attn_input_size, 1, bias=False, device=device)
        self.edge_attn = Linear(attn_input_size, 1, bias=False, device=device)

        self.node_tgt_attn = Linear((hidden_size * 2), 1, bias=False, device=device)
        self.node_src_attn = Linear((hidden_size * 2), 1, bias=False, device=device)

        self.msg_mlp = make_mlp(input_size=(node_in_channels + hidden_size * 2), output_size=edge_out_channels,
                                hidden_size=hidden_size * 2, num_layers=num_layers,
                                activation=activation, device=device)

        node_update_input_size = (edge_out_channels)
        node_update_hidden_size = node_update_input_size * 2
        self.node_update_mlp = make_mlp(input_size=node_update_input_size, output_size=node_out_channels,
                                 hidden_size=node_update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

        edge_update_input_size = (edge_in_channels + node_out_channels * 2)
        edge_update_hidden_size = edge_update_input_size * 2
        self.edge_update_mlp = make_mlp(input_size=edge_update_input_size, output_size=edge_out_channels,
                                 hidden_size=edge_update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        alpha, beta, delta, gamma = self.compute_attention(edge_index, x=x, edge_attr=edge_attr)
        x, edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr, alpha=alpha, beta=beta, delta=delta, gamma=gamma)
        return x, edge_attr

    def compute_attention(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._attn_user_args, edge_index, mutable_size, kwargs)
        edge_kwargs = self.inspector.collect_param_data('attention', coll_dict)
        attn_weights = self.attention(**edge_kwargs)
        return attn_weights

    def attention(self, x_i, x_j, edge_attr, index, edge_index_j, ptr, dim_size) -> Tensor:
        alpha = self.node_attn(torch.concat([x_i, edge_attr, x_j], dim=-1))
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, dim_size)

        beta = self.edge_attn(torch.concat([x_i, edge_attr, x_j], dim=-1))
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = softmax(beta, index, ptr, dim_size)

        gamma = self.node_tgt_attn(torch.concat([x_i, edge_attr], dim=-1))
        gamma = F.leaky_relu(gamma, negative_slope=0.2)
        gamma = softmax(gamma, index, ptr, dim_size)

        delta = self.node_src_attn(torch.concat([x_j, edge_attr], dim=-1))
        delta = F.leaky_relu(delta, negative_slope=0.2)
        delta = softmax(delta, edge_index_j, ptr, dim_size)

        return alpha, beta, gamma, delta

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggr = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        node_out = self.update(aggr, **update_kwargs)

        coll_dict = self._collect(self._edge_user_args, edge_index, mutable_size, {'node_out': node_out, 'msg': msg, **kwargs})
        edge_update_kwargs = self.inspector.collect_param_data('edge_update', coll_dict)
        edge_out = self.edge_update(**edge_update_kwargs)

        return node_out, edge_out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, alpha: Tensor, beta: Tensor):
        return self.msg_mlp(torch.cat([x_i, alpha * x_j, beta * edge_attr], dim=-1))

    def update(self, aggr: Tensor, x: Tensor):
        node_update = self.node_update_mlp(aggr)
        return x + node_update

    def edge_update(self, msg, edge_attr, node_out_i: Tensor, node_out_j: Tensor, gamma: Tensor, delta: Tensor):
        edge_update = self.edge_update_mlp(torch.cat([msg, (node_out_i * gamma), (node_out_j * delta)], dim=-1))
        return edge_attr + edge_update
