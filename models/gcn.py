from utils.model_utils import make_gnn 

from .base_node_model import BaseNodeModel

class GCN(BaseNodeModel):
    '''
    GCN (Graph Convolutional Network)
    '''
    def __init__(self, **base_model_kwargs):
        super().__init__(
            use_edge_features=False,
            **base_model_kwargs
        )

        self.convs = make_gnn(input_size=self.input_size, output_size=self.output_size,
                              hidden_size=self.hidden_features, num_layers=self.num_layers,
                              conv='gcn', activation=self.activation, device=self.device)

