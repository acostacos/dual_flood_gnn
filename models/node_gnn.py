from torch import Tensor

from .node_edge_gnn import NodeEdgeGNN

class NodeGNN(NodeEdgeGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, _ = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)

        return x
