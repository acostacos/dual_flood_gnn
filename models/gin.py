from torch_geometric.nn import GINConv, Sequential as PygSequential
from utils.model_utils import make_mlp

from .base_node_model import BaseNodeModel

class GIN(BaseNodeModel):
    '''
    GIN (Graph Isomorphism Network)
    '''
    def __init__(self,
                 mlp_layers: int = 2,
                 eps: float = 0.0,
                 train_eps: bool = False,
                 **base_model_kwargs):
        super().__init__(
            use_edge_features=False,
            **base_model_kwargs
        )

        self.convs = self._make_gnn(input_size=self.input_size, output_size=self.output_size,
                                    hidden_size=self.hidden_features, num_layers=self.num_layers, mlp_layers=mlp_layers,
                                    activation=self.activation, device=self.device, eps=eps, train_eps=train_eps)

    def _make_gnn(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, mlp_layers: int,
                  activation: str, device: str, eps: float, train_eps: bool):
        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)
        conv_kwargs = {
            'hidden_size': hidden_size,
            'mlp_layers': mlp_layers,
            'activation': activation,
            'device': device,
            'eps': eps,
            'train_eps': train_eps
        }

        if num_layers == 1:
            return self._get_gin_layer(**conv_kwargs)

        layers = []
        layers.append(
            (self._get_gin_layer(input_size, hidden_size, **conv_kwargs), 'x, edge_index -> x')
        ) # Input Layer
        for _ in range(num_layers-2):
            layers.append(
                (self._get_gin_layer(hidden_size, hidden_size, **conv_kwargs), 'x, edge_index -> x')
            ) # Hidden Layers
        layers.append(
            (self._get_gin_layer(hidden_size, output_size, **conv_kwargs), 'x, edge_index -> x')
        ) # Output Layer
        return PygSequential('x, edge_index', layers)

    def _get_gin_layer(self, input_size: int, output_size: int, hidden_size: int, mlp_layers: int, activation: str,
                       device: str, eps: float, train_eps: bool) -> GINConv:
        mlp = make_mlp(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=mlp_layers,
                       activation=activation, device=device)
        return GINConv(nn=mlp, eps=eps, train_eps=train_eps).to(device)
