import torch

from torch import Tensor
from torch.nn import Identity
from typing import Optional
from utils.model_utils import make_mlp

from .base_model import BaseModel


class BaseEdgeModel(BaseModel):
    '''
    Base class for edge prediction models.
    '''
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 use_edge_features: bool = False,
                 input_edge_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 residual: bool = False,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        assert decoder_layers > 0, f"{self.__class__.__name__} requires a decoder to map node embeddings to edge outputs."

        self.with_encoder = encoder_layers > 0
        self.hidden_features = hidden_features
        self.use_edge_features = use_edge_features
        self.num_layers = num_layers
        self.activation = activation

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_edge_features
        if self.use_edge_features and input_edge_features is None:
            input_edge_features = self.input_edge_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                         hidden_size=hidden_features, num_layers=encoder_layers,
                                         activation=encoder_activation, device=self.device)
            if self.use_edge_features:
                self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                             hidden_size=hidden_features, num_layers=encoder_layers,
                                             activation=encoder_activation, device=self.device)

        # Processor
        self.input_size = hidden_features if self.with_encoder else input_features
        self.output_size = hidden_features
        self.input_edge_size = None
        if self.use_edge_features:
            self.input_edge_size = hidden_features if self.with_encoder else input_edge_features
        self.convs = None  # To be defined in child classes

        # Decoder
        decoder_input_size = 2 * hidden_features
        self.edge_decoder = make_mlp(input_size=decoder_input_size, output_size=output_features,
                                     hidden_size=hidden_features, num_layers=decoder_layers,
                                     activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        edge_attr0 = edge_attr.clone() if edge_attr is not None else None

        if self.with_encoder:
            x = self.node_encoder(x)
            if self.use_edge_features and edge_attr is not None:
                edge_attr = self.edge_encoder(edge_attr)

        if self.use_edge_features:
            x = self.convs(x, edge_index, edge_attr)
        else:
            x = self.convs(x, edge_index)

        row, col = edge_index
        edge_attr = self.edge_decoder(torch.cat([x[row], x[col]], dim=-1))

        if hasattr(self, 'residual') and edge_attr0 is not None:
            edge_attr = edge_attr + self.residual(edge_attr0[:, -self.output_node_features:])

        return edge_attr
