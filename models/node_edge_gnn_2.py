import torch
from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class NodeEdgeGNN2(BaseModel):
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
            return NodeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                                num_layers=mlp_layers, activation=activation, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append((EdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                             'x, edge_index, edge_attr -> x, edge_attr'))
            layers.append((NodeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                             'x, edge_index, edge_attr -> x, edge_attr'))
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

class NodeConv(MessagePassing):
    """
    Message = MLP between edge and node
    Aggregate = sum
    Update = MLP between node and aggr
    """
    def __init__(self, node_in_channels: int, node_out_channels: int, edge_in_channels: int, edge_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        msg_input_size = (node_in_channels + edge_in_channels) # Neighboring node + edge features
        msg_hidden_size = msg_input_size * 2
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out_channels,
                            hidden_size=msg_hidden_size, num_layers=num_layers,
                            activation=activation, device=device)

        update_input_size = (node_in_channels + edge_out_channels) # Node features + aggregated edge features
        update_hidden_size = update_input_size * 2
        self.update_mlp = make_mlp(input_size=update_input_size, output_size=node_out_channels,
                                 hidden_size=update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        out = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(out, **update_kwargs)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor):
        return self.msg_mlp(torch.cat([edge_attr, x_j], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))

class EdgeConv(MessagePassing):
    """
    Message = MLP between node and node
    Aggregate = sum
    Update = MLP between edge and aggr
    """
    def __init__(self, node_in_channels: int, node_out_channels: int, edge_in_channels: int, edge_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        msg_input_size = (node_in_channels * 2) # Neighboring node features (i, j)
        msg_hidden_size = msg_input_size * 2
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out_channels,
                            hidden_size=msg_hidden_size, num_layers=num_layers,
                            activation=activation, device=device)

        update_input_size = (edge_in_channels + edge_out_channels) # Edge features + node messages
        update_hidden_size = update_input_size * 2
        self.update_mlp = make_mlp(input_size=update_input_size, output_size=edge_out_channels,
                                 hidden_size=update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def edge_updater(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        out = self.edge_update(msg, kwargs['edge_attr'])

        return out

    def message(self, x_j: Tensor, x_i: Tensor):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))

    def edge_update(self, aggr_out: Tensor, edge_attr: Tensor):
        return self.update_mlp(torch.cat([edge_attr, aggr_out], dim=-1))
