from utils.model_utils import make_gnn 

from .base_edge_model import BaseEdgeModel

class EdgeGCN(BaseEdgeModel):
    '''
    GCN (Graph Convolutional Network)
    Modified for Edge Prediction.
    '''
    def __init__(self, **base_model_kwargs):
        super().__init__(
            use_edge_features=False,
            **base_model_kwargs
        )

        self.convs = make_gnn(input_size=self.input_size, output_size=self.output_size,
                              hidden_size=self.hidden_features, num_layers=self.num_layers,
                              conv='gcn', activation=self.activation, device=self.device)
