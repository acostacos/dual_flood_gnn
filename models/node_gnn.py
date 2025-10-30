from torch import Tensor

from .dual_flood_gnn import DUALFloodGNN

class NodeGNN(DUALFloodGNN):
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
