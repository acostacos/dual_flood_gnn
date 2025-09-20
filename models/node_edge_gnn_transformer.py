import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, Linear, MultiheadAttention, Dropout, Sequential as TorchSequential
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from torch_geometric.nn.inits import reset
from torch_geometric.utils import to_dense_batch
from typing import Tuple, Optional, Dict, Any
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class NodeEdgeGNNTransformer(BaseModel):
    '''
    NodeEdgeGNN with Transformer
    '''
    def __init__(self,
                 input_features: int = None,
                 input_edge_features: int = None,
                 output_features: int = None,
                 output_edge_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 2,
                 activation: str = 'relu',
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

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        if input_edge_features is None:
            input_edge_features = self.input_edge_features
        if output_edge_features is None:
            output_edge_features = self.output_edge_features

        # Encoder
        encoder_decoder_hidden = hidden_features*2
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                         hidden_size=encoder_decoder_hidden, num_layers=encoder_layers,
                                         activation=encoder_activation, bias=False, device=self.device)
            self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                         hidden_size=hidden_features, num_layers=encoder_layers,
                                         activation=encoder_activation, bias=False, device=self.device)

        input_node_size = hidden_features if self.with_encoder else input_features
        output_node_size = hidden_features if self.with_decoder else output_features
        input_edge_size = hidden_features if self.with_encoder else input_edge_features
        output_edge_size = hidden_features if self.with_decoder else output_edge_features

        self.convs = self._make_gnn(input_node_size=input_node_size, output_node_size=output_node_size,
                                    input_edge_size=input_edge_size, output_edge_size=output_edge_size,
                                    hidden_features=hidden_features, num_layers=num_layers, mlp_layers=mlp_layers,
                                    activation=activation, residual=residual, device=self.device)
        self.convs = self.convs.to(self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=output_edge_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  hidden_features: int, num_layers: int, mlp_layers: int, activation: str, residual: bool, device: str):
        if num_layers == 1:
            mpnn = NodeEdgeConv(node_in_channels=input_node_size, edge_in_channels=input_edge_size,
                                node_out_channels=output_node_size, edge_out_channels=output_edge_size,
                                hidden_size=hidden_features, num_layers=mlp_layers, activation=activation,
                                residual=residual, bias=False, device=device)
            conv = (
                GPSWithEdgeLayer(input_node_size, mpnn, heads=1, attn_type='multihead', attn_kwargs={}),
                'x, edge_index, edge_attr -> x, edge_attr',
            )
            return conv

        layers = []
        for _ in range(num_layers):
            mpnn = NodeEdgeConv(node_in_channels=input_node_size, edge_in_channels=input_edge_size,
                                node_out_channels=output_node_size, edge_out_channels=output_edge_size,
                                hidden_size=hidden_features, num_layers=mlp_layers, activation=activation,
                                residual=residual, bias=False, device=device)
            conv = (
                GPSWithEdgeLayer(input_node_size, mpnn, heads=1, attn_type='multihead', attn_kwargs={}),
                'x, edge_index, edge_attr -> x, edge_attr',
            )

            layers.append(conv)
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        return x, edge_attr

class GPSWithEdgeLayer(Module):
    '''
    Based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gps_conv.py
    '''
    def __init__(
        self,
        channels: int,
        conv: MessagePassing,
        heads: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            **attn_kwargs,
        )

        self.mlp = TorchSequential(
            Linear(channels, channels * 2),
            get_activation_func(activation),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        # self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        # self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        # self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        # if self.norm1 is not None:
        #     self.norm1.reset_parameters()
        # if self.norm2 is not None:
        #     self.norm2.reset_parameters()
        # if self.norm3 is not None:
        #     self.norm3.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Optional[torch.Tensor] = None) -> Tensor:
        hs = []
        # Local MPNN.
        h, edge_attr = self.conv(x, edge_index, edge_attr)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        # if self.norm1 is not None:
        #     if self.norm_with_batch:
        #         h = self.norm1(h, batch=batch)
        #     else:
        #         h = self.norm1(h)
        hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch) # Convert all graphs in the batch into the same size.
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        # if self.norm2 is not None:
            # if self.norm_with_batch:
            #     h = self.norm2(h, batch=batch)
            # else:
            #     h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        # if self.norm3 is not None:
        #     if self.norm_with_batch:
        #         out = self.norm3(out, batch=batch)
        #     else:
        #         out = self.norm3(out)

        return out, edge_attr

class NodeEdgeConv(MessagePassing):
    '''
    Message = MLP(node_i, edge_attr, node_j)
    Aggregate = sum
    Update = MLP(aggregated_message)
    '''
    def __init__(self, node_in_channels: int, edge_in_channels: int,
                 node_out_channels: int, edge_out_channels: int,
                 hidden_size: int, num_layers: int = 2, activation: str = 'relu',
                 residual: bool = True, bias: bool = False, device: str = 'cpu'):
        super().__init__(aggr='sum')
        self.residual = residual

        input_size = node_in_channels
        output_size = node_out_channels
        self.node_mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                                 num_layers=num_layers, activation=activation, bias=bias, device=device)

        input_size = (node_in_channels * 2 + edge_in_channels)
        output_size = edge_out_channels
        self.edge_mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                                 num_layers=num_layers, activation=activation, bias=bias, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggr = self.aggregate(msg, **aggr_kwargs)

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
