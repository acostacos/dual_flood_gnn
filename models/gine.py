from torch import Tensor
from torch_geometric.nn import GINEConv, Sequential as PygSequential
from typing import Optional
from utils.model_utils import make_mlp

from .base_model import BaseModel

class GINE(BaseModel):
    '''
    GIN with edge features.
    '''
    def __init__(self,
                 input_features: int = None,
                 input_edge_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 mlp_layers: int = 2,

                 # Convolution Parameters
                 eps: float = 0.0,
                 train_eps: bool = False,

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
        if input_edge_features is None:
            input_edge_features = self.input_edge_features
        if output_features is None:
            output_features = self.output_node_features

        input_size = hidden_features if self.with_encoder else input_features
        input_edge_size = hidden_features if self.with_encoder else input_edge_features
        output_size = hidden_features if self.with_decoder else output_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                         hidden_size=hidden_features, num_layers=encoder_layers,
                                         activation=encoder_activation, device=self.device)
            self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                         hidden_size=hidden_features, num_layers=encoder_layers,
                                         activation=encoder_activation, device=self.device)

        self.convs = self._make_gnn(input_size=input_size, edge_input_size=input_edge_size, output_size=output_size,
                                    hidden_size=hidden_features, num_layers=num_layers, mlp_layers=mlp_layers,
                                    activation=activation, device=self.device, eps=eps, train_eps=train_eps)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                         hidden_size=hidden_features, num_layers=encoder_layers,
                                         activation=decoder_activation, bias=False, device=self.device)

    def _make_gnn(self, input_size: int, edge_input_size: int, output_size: int, hidden_size: int, num_layers: int,
                  mlp_layers: int, activation: str, device: str, eps: float, train_eps: bool):
        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)
        conv_kwargs = {
            'hidden_size': hidden_size,
            'mlp_layers': mlp_layers,
            'activation': activation,
            'device': device,
            'eps': eps,
            'train_eps': train_eps,
            'edge_dim': edge_input_size,
        }

        if num_layers == 1:
            return self._get_gine_layer(**conv_kwargs)

        layers = []
        layers.append(
            (self._get_gine_layer(input_size, hidden_size, **conv_kwargs), 'x, edge_index, edge_attr -> x')
        ) # Input Layer
        for _ in range(num_layers-2):
            layers.append(
                (self._get_gine_layer(hidden_size, hidden_size, **conv_kwargs), 'x, edge_index, edge_attr -> x')
            ) # Hidden Layers
        layers.append(
            (self._get_gine_layer(hidden_size, output_size, **conv_kwargs), 'x, edge_index, edge_attr -> x')
        ) # Output Layer
        return PygSequential('x, edge_index, edge_attr', layers)

    def _get_gine_layer(self, input_size: int, output_size: int, hidden_size: int, mlp_layers: int, activation: str,
                       device: str, eps: float, train_eps: bool, edge_dim: int) -> GINEConv:
        mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=mlp_layers,
                       activation=activation, device=device)
        return GINEConv(nn=mlp, eps=eps, train_eps=train_eps, edge_dim=edge_dim).to(device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        if self.with_encoder:
            x = self.node_encoder(x)

        x = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)

        return x
