import math
import torch
import torch.nn.functional as F

from torch.nn import Parameter, ModuleList
from torch_geometric.nn import MessagePassing

from .base_model import BaseModel

class CensNet(BaseModel):
    """
    Co-embedding of Nodes and Edges with Graph Neural Networks
    Implemented based on https://github.com/ronghangzhu/CensNet
    """
    def __init__(self,
                 input_features: int = None,
                 input_edge_features: int = None,
                 output_features: int = None,
                 output_edge_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 bias: bool = True,
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.dropout = dropout

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        if input_edge_features is None:
            input_edge_features = self.input_edge_features
        if output_edge_features is None:
            output_edge_features = self.output_edge_features

        self.node_convs, self.edge_convs = self._make_gnn(in_node=input_features, out_node=output_features,
                                                          in_edge=input_edge_features, out_edge=output_edge_features,
                                                          hid=hidden_features, num_layers=num_layers, bias=bias)
        self.node_convs, self.edge_convs = self.node_convs.to(self.device), self.edge_convs.to(self.device)

    def _make_gnn(self, in_node, out_node, in_edge, out_edge, hid, num_layers, bias):
        node_convs = ModuleList()
        edge_convs = ModuleList()

        node_convs.append(GraphConvolution(in_node, hid, in_edge, in_edge, bias, node_layer=True))
        edge_convs.append(GraphConvolution(hid, hid, in_edge, hid, bias, node_layer=False))
        for _ in range(num_layers-2):
            node_convs.append(GraphConvolution(hid, hid, hid, hid, bias, node_layer=True))
            edge_convs.append(GraphConvolution(hid, hid, hid, hid, bias, node_layer=False))
        node_convs.append(GraphConvolution(hid, out_node, hid, hid, bias, node_layer=True))
        edge_convs.append(GraphConvolution(hid, hid, hid, out_edge, bias, node_layer=False))

        return node_convs, edge_convs

    def forward(self, x, edge_index, edge_attr, dual_edge_index, dual_edge_attr):
        for conv_v, conv_e in zip(self.node_convs[:-1], self.edge_convs[:-1]):
            x, edge_attr = conv_v(x, edge_index, edge_attr, dual_edge_index, dual_edge_attr)
            x, edge_attr = F.relu(x), F.relu(edge_attr)

            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

            x, edge_attr = conv_e(x, edge_index, edge_attr, dual_edge_index, dual_edge_attr)
            x, edge_attr = F.relu(x), F.relu(edge_attr)

            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        out_x, _ = self.node_convs[-1](x, edge_index, edge_attr, dual_edge_index, dual_edge_attr)
        _, out_edge_attr = self.edge_convs[-1](x, edge_index, edge_attr, dual_edge_index, dual_edge_attr)

        return out_x, out_edge_attr

class GraphConvolution(MessagePassing):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, 
                 bias=True, node_layer=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.node_layer = node_layer
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e

        if node_layer:
            # Node layer: transform nodes, modulate by edge features
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.FloatTensor(1, in_features_e))
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            # Edge layer: transform edges, modulate by node features
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.FloatTensor(1, in_features_v))
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, dual_edge_index, dual_edge_attr):
        if self.node_layer:
            edge_weights = (edge_attr @ self.p.t()).squeeze(-1)
            x_transformed = x @ self.weight

            out = self.propagate(edge_index, x=x_transformed, edge_weight=edge_weights)

            if self.bias is not None:
                out = out + self.bias
            return out, edge_attr
        else:
            node_weights = (x @ self.p.t()).squeeze(-1)
            dual_edge_weights = node_weights[dual_edge_attr]
            edge_transformed = edge_attr @ self.weight

            out = self.propagate(dual_edge_index, x=edge_transformed, edge_weight=dual_edge_weights)

            if self.bias is not None:
                out = out + self.bias
            return x, out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv_p = 1. / math.sqrt(self.p.size(1))
        self.p.data.uniform_(-stdv_p, stdv_p)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
