import torch
from torch import Tensor
from torch.nn import Identity
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from typing import Tuple
from utils.model_utils import make_mlp

from .base_model import BaseModel

class HydroGraphNet(BaseModel):
    def __init__(self,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
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


        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, norm='layernorm', bias=False, device=self.device)
            self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, norm='layernorm', bias=False, device=self.device)

        input_node_size = hidden_features if self.with_encoder else self.input_node_features
        output_node_size = hidden_features if self.with_decoder else self.output_node_features
        input_edge_size = hidden_features if self.with_encoder else self.input_edge_features
        output_edge_size = hidden_features if self.with_decoder else self.output_edge_features
        self.node_convs = [
            MeshNodeBlock(node_in_channels=input_node_size, node_out_channels=output_node_size,
                          edge_in_channels=input_edge_size, hidden_size=hidden_features,
                          num_layers=mlp_layers, activation=activation, norm='layernorm', bias=False, device=self.device)
            for _ in range(num_layers)
        ]
        self.edge_convs = [
            MeshEdgeBlock(node_in_channels=input_node_size, edge_in_channels=input_edge_size,
                          edge_out_channels=output_edge_size, hidden_size=hidden_features,
                          num_layers=mlp_layers, activation=activation, norm='layernorm', bias=False, device=self.device)
            for _ in range(num_layers)
        ]

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=hidden_features, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=self.output_edge_features,
                                        hidden_size=hidden_features, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        for edge_conv, node_conv in zip(self.edge_convs, self.node_convs):
            x, edge_attr = edge_conv(x, edge_index, edge_attr)
            x, edge_attr = node_conv(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            # edge_attr = self.edge_decoder(edge_attr)

        # return x, edge_attr
        return x

class MeshEdgeBlock(MessagePassing):
    def __init__(self, node_in_channels: int, edge_in_channels: int, edge_out_channels: int, hidden_size: int = None,
                 num_layers: int = 2, activation: str = 'relu', norm: str = 'layernorm', bias: bool = False, device: str = 'cpu'):
        super().__init__(aggr='sum')
        input_size = (node_in_channels * 2 + edge_in_channels)
        output_size = edge_out_channels
        self.edge_mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                                 num_layers=num_layers, activation=activation, norm=norm, bias=bias, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def edge_update(self, edge_attr: Tensor, x_i: Tensor, x_j: Tensor) -> Tensor:
        cat_feats = torch.cat([edge_attr, x_i, x_j], dim=-1)
        return self.edge_mlp(cat_feats) + edge_attr

class MeshNodeBlock(MessagePassing):
    def __init__(self, node_in_channels: int, edge_in_channels: int, node_out_channels: int, hidden_size: int = None,
                 num_layers: int = 2, activation: str = 'relu', norm: str = 'layernorm', bias: bool = False, device: str = 'cpu'):
        super().__init__(aggr='sum')
        input_size = (node_in_channels + edge_in_channels)
        output_size = node_out_channels
        self.node_mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                                 num_layers=num_layers, activation=activation, norm=norm, bias=bias, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return torch.cat([edge_attr, x_j], dim=-1)

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        return self.node_mlp(aggr) + x
