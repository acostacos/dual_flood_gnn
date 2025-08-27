from torch import Tensor
from torch.nn import Identity
from typing import Optional
from utils.model_utils import make_mlp, make_gnn 

from .base_model import BaseModel

class GCN(BaseModel):
    '''
    GCN
    Most Basic Graph Neural Network w/ Encoder-Decoder
    '''
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,

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

        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        self.convs = make_gnn(input_size=input_size, output_size=output_size,
                              hidden_size=hidden_features, num_layers=num_layers,
                              conv='gcn', activation=activation, device=self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        x0 = x.clone()

        if self.with_encoder:
            x = self.node_encoder(x)

        x = self.convs(x, edge_index)

        if self.with_decoder:
            x = self.node_decoder(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x

