import torch

from data import FloodEventDataset
from data.dataset_normalizer import DatasetNormalizer
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter
from typing import Dict, Tuple

from .loss_helper import get_orig_water_volume, get_orig_water_flow

class LocalMassConservationLoss(Module):
    def __init__(self,
                 previous_timesteps: int,
                 normalizer: DatasetNormalizer,
                 is_normalized: bool = True,
                 delta_t: int = 30):
        super(LocalMassConservationLoss, self).__init__()
        self.previous_timesteps = previous_timesteps
        self.normalizer = normalizer
        self.is_normalized = is_normalized
        self.delta_t = delta_t

    def forward(self,
                batch_node_pred: Tensor, # Normalized predicted water volume (t+1)
                batch_edge_input: Tensor, # Normalized given water flow w/ unmasked outflow (t)
                databatch) -> Tensor:
        batch = databatch.batch
        edge_index = databatch.edge_index
        num_nodes = databatch.num_nodes
        num_graphs = databatch.num_graphs
        local_mass_info: Dict[str, Tensor] = databatch.local_mass_info

        # Get predefined information
        rainfall = local_mass_info['rainfall']
        non_boundary_nodes_mask = local_mass_info['non_boundary_nodes_mask']

        # Get current total water volume (t)
        curr_water_volume = self.get_curr_volume_from_batch(databatch, non_boundary_nodes_mask) # Normalized given water volume (t)

        # Get next total water volume (t+1)
        next_water_volume = get_orig_water_volume(batch_node_pred, self.normalizer, self.is_normalized, non_boundary_nodes_mask)

        # Get current water flow (t)
        water_flow = get_orig_water_flow(batch_edge_input, self.normalizer, self.is_normalized)
        total_inflow, total_outflow = self.get_batch_inflow_outflow(edge_index, water_flow, non_boundary_nodes_mask, num_nodes)

        # Compute Local Mass Conservation
        delta_v = next_water_volume - curr_water_volume
        rf_volume = rainfall
        inflow_volume =  total_inflow * self.delta_t
        outflow_volume = total_outflow * self.delta_t

        local_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        local_volume_error = torch.abs(local_volume_error)
        non_boundary_batch = batch[non_boundary_nodes_mask]
        total_local_volume_error = scatter(local_volume_error, non_boundary_batch, reduce='sum', dim_size=num_graphs)

        local_loss = total_local_volume_error.mean()
        return local_loss

    def get_curr_volume_from_batch(self, databatch, non_boundary_nodes_mask: Tensor) -> Tensor:
        water_volume_dyn_num = FloodEventDataset.DYNAMIC_NODE_FEATURES.index('water_volume') + 1
        num_static_node_features = len(FloodEventDataset.STATIC_NODE_FEATURES)
        curr_water_volume_idx = num_static_node_features + ((self.previous_timesteps + 1) * water_volume_dyn_num) - 1
        curr_water_volume = databatch.x[:, [curr_water_volume_idx]]

        curr_water_volume = get_orig_water_volume(curr_water_volume, self.normalizer, self.is_normalized, non_boundary_nodes_mask)
        return curr_water_volume

    def get_batch_inflow_outflow(self,
                                 edge_index: Tensor,
                                 face_flow: Tensor,
                                 non_boundary_nodes_mask: ndarray,
                                 num_nodes: int) -> Tuple[Tensor, Tensor]:
        # Convert edge_index and face_flow to undirected w/ flipped edge features for inflow/outflow calculations
        row, col = edge_index[0], edge_index[1]
        row, col = torch.cat([row, col], axis=0), torch.cat([col, row], axis=0)
        edge_index = torch.stack([row, col], axis=0)
        face_flow = torch.cat([face_flow, -face_flow], axis=0)

        face_flow = torch.relu(face_flow) # Negative flow would just be opposite direction of positive flow; Can be ignored
        total_inflow = scatter(face_flow, edge_index[1], reduce='sum', dim_size=num_nodes)
        total_inflow = total_inflow[non_boundary_nodes_mask]
        total_outflow = scatter(face_flow, edge_index[0], reduce='sum', dim_size=num_nodes)
        total_outflow = total_outflow[non_boundary_nodes_mask]

        return total_inflow, total_outflow
