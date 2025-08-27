from torch import Tensor
from torch.nn import Module

class BaseModel(Module):
    def __init__(self,
                 static_node_features: int = 0,
                 dynamic_node_features: int = 0,
                 static_edge_features: int = 0,
                 dynamic_edge_features: int = 0,
                 previous_timesteps: int = 0,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.previous_timesteps = previous_timesteps

        self.static_node_features = static_node_features
        self.dynamic_node_features = dynamic_node_features
        self.input_node_features = self.static_node_features + (self.dynamic_node_features * (self.previous_timesteps+1))
        self.output_node_features = 1 # Water Level

        self.static_edge_features = static_edge_features
        self.dynamic_edge_features = dynamic_edge_features
        self.input_edge_features = self.static_edge_features + (self.dynamic_edge_features * (self.previous_timesteps+1))
        self.output_edge_features = 1 # Velocity

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented!")

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
