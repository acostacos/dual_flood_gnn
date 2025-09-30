import torch

from data.dataset_normalizer import DatasetNormalizer
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter
from typing import Dict, Tuple, Literal, Optional

from .loss_helper import get_orig_water_volume, get_orig_water_flow

class LocalMassConservationLoss(Module):
    '''
    Implements local mass conservation loss. Behavior changes depending on mode (train/test).
    During training, we take the absolute value of the loss to convert it to a convex function.
    During testing, we return the original signed values.
    '''
    def __init__(self,
                 mode: Literal['train', 'test'],
                 normalizer: DatasetNormalizer,
                 is_normalized: bool = True,
                 delta_t: int = 30):
        super(LocalMassConservationLoss, self).__init__()
        self.mode = mode
        self.normalizer = normalizer
        self.is_normalized = is_normalized
        self.delta_t = delta_t

    def forward(self,
                batch_node_pred: Tensor, # Normalized predicted water volume (t+1)
                batch_node_input: Tensor, # Normalized given water volume (t)
                batch_edge_input: Tensor, # Normalized given water flow w/ unmasked outflow (t)
                rainfall: Tensor, # Actual rainfall (not normalized), from local_mass_info
                databatch,
                node_filter_mask: Optional[Tensor] = None) -> Tensor:
        batch = databatch.batch
        edge_index = databatch.edge_index
        num_nodes = databatch.num_nodes
        num_graphs = databatch.num_graphs
        local_mass_info: Dict[str, Tensor] = databatch.local_mass_info

        if node_filter_mask is None:
            # Default to using all non-boundary nodes
            node_filter_mask = local_mass_info['non_boundary_nodes_mask']

        # Get current total water volume (t)
        curr_water_volume = get_orig_water_volume(batch_node_input, self.normalizer, self.is_normalized, node_filter_mask)

        # Get next total water volume (t+1)
        next_water_volume = get_orig_water_volume(batch_node_pred, self.normalizer, self.is_normalized, node_filter_mask)

        # Get current water flow (t)
        water_flow = get_orig_water_flow(batch_edge_input, self.normalizer, self.is_normalized)
        total_inflow, total_outflow = self.get_batch_inflow_outflow(edge_index, water_flow, node_filter_mask, num_nodes)

        rainfall = rainfall[node_filter_mask]

        # Compute Local Mass Conservation
        delta_v = next_water_volume - curr_water_volume
        rf_volume = rainfall
        inflow_volume =  total_inflow * self.delta_t
        outflow_volume = total_outflow * self.delta_t

        local_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume

        if self.mode == 'train':
            local_volume_error = torch.abs(local_volume_error)

        non_boundary_batch = batch[node_filter_mask]
        total_local_volume_error = scatter(local_volume_error, non_boundary_batch, reduce='sum', dim_size=num_graphs)

        if self.mode == 'train':
            local_loss = total_local_volume_error.mean()
        else:
            local_loss = total_local_volume_error.sum()

        return local_loss

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
