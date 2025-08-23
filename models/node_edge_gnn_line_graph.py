import torch

from data.line_graph_data import LineGraphData
from torch import Tensor
from torch.nn import Identity, ModuleList
from torch_geometric.nn import MessagePassing, GATConv
from utils.model_utils import make_mlp

from .base_model import BaseModel

class NodeEdgeGNNLineGraph(BaseModel):
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

        self.node_convs, self.edge_convs = self._make_gnn(input_node_size=input_size, output_node_size=output_size,
                                    input_edge_size=input_edge_size, output_edge_size=output_edge_size,
                                    num_layers=num_layers, mlp_layers=attn_mlp_layers, activation=activation, device=self.device)
        self.node_convs, self.edge_convs = self.node_convs.to(self.device), self.edge_convs.to(self.device)

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

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  num_layers: int, mlp_layers: int, activation: str, device: str):
        node_convs = ModuleList()
        edge_convs = ModuleList()
        for _ in range(num_layers):
            # node_convs.append(NodeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
            #                  num_layers=mlp_layers, activation=activation, device=device))
            # edge_convs.append(EdgeConv(input_edge_size, output_edge_size, input_node_size, output_node_size,
            #                  num_layers=mlp_layers, activation=activation, device=device))
            node_convs.append(GATConv(in_channels=input_node_size, out_channels=output_node_size,
                                      edge_dim=input_edge_size, add_self_loops=False))
            edge_convs.append(GATConv(in_channels=input_edge_size, out_channels=output_edge_size,
                                      edge_dim=input_node_size, add_self_loops=False))
        return node_convs, edge_convs

    def forward(self, graph: LineGraphData) -> Tensor:
        # x, edge_index, edge_attr, line_edge_index, line_edge_attr_idx = graph.x, graph.edge_index, graph.edge_attr, graph.line_edge_index, graph.line_edge_attr
        # x0, edge_attr0 = x.clone(), edge_attr.clone()
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        line_edge_index, line_edge_attr_idx = graph.line_edge_index.clone(), graph.line_edge_attr.clone()
        x0, edge_attr0 = x, edge_attr

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        for node_conv, edge_conv in zip(self.node_convs, self.edge_convs):
            x = node_conv(x, edge_index, edge_attr)

            line_edge_attr = x[line_edge_attr_idx]
            edge_attr = edge_conv(edge_attr, line_edge_index, line_edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])
            edge_attr = edge_attr + self.residual(edge_attr0[:, -self.output_edge_features:])

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
        return x

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
    Edges = Nodes
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
        return x

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
