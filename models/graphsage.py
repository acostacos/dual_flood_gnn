from utils.model_utils import make_gnn 

from .base_node_model import BaseNodeModel
from .base_edge_model import BaseEdgeModel

class GraphSAGE(BaseNodeModel):
    '''
    GraphSAGE (Graph Sample and Aggregate)
    '''
    def __init__(self,
                 aggr: str = 'mean',
                 normalize: bool = False,
                 root_weight: bool = True,
                 project: bool = False,
                 bias: bool = True,
                 **base_model_kwargs):
        super().__init__(
            use_edge_features=False,
            **base_model_kwargs
        )

        self.convs = make_gnn(input_size=self.input_size, output_size=self.output_size,
                              hidden_size=self.hidden_features, num_layers=self.num_layers,
                              conv='sage', activation=self.activation, device=self.device,
                              aggr=aggr, normalize=normalize, root_weight=root_weight,
                              project=project, bias=bias)

class EdgeGraphSAGE(BaseEdgeModel):
    '''
    GraphSAGE (Graph Sample and Aggregate)
    Modified for Edge Prediction.
    '''
    def __init__(self,
                 aggr: str = 'mean',
                 normalize: bool = False,
                 root_weight: bool = True,
                 project: bool = False,
                 bias: bool = True,
                 **base_model_kwargs):
        super().__init__(
            use_edge_features=False,
            **base_model_kwargs
        )

        self.convs = make_gnn(input_size=self.input_size, output_size=self.output_size,
                              hidden_size=self.hidden_features, num_layers=self.num_layers,
                              conv='sage', activation=self.activation, device=self.device,
                              aggr=aggr, normalize=normalize, root_weight=root_weight,
                              project=project, bias=bias)
