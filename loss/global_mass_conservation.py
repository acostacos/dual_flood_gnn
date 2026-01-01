import torch

from data.dataset_normalizer import DatasetNormalizer
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter
from typing import Dict, Literal

from .loss_helper import get_orig_water_volume, get_orig_water_flow

class GlobalMassConservationLoss(Module):
    '''
    Implements global mass conservation loss. Behavior changes depending on mode (train/test).
    During training, we take the absolute value of the loss to convert it to a convex function.
    During testing, we return the original signed values.
    '''
    def __init__(self,
                 mode: Literal['train', 'test'],
                 normalizer: DatasetNormalizer,
                 is_normalized: bool = True,
                 delta_t: int = 30):
        super(GlobalMassConservationLoss, self).__init__()
        self.mode = mode
        self.normalizer = normalizer
        self.is_normalized = is_normalized
        self.delta_t = delta_t

    def forward(self,
                batch_node_pred: Tensor, # Normalized predicted water volume (t+1)
                batch_node_input: Tensor, # Normalized given water volume (t)
                batch_edge_input: Tensor, # Normalized given water flow w/ unmasked outflow (t)
                total_rainfall: Tensor, # Actual total rainfall (not normalized), from global_mass_info
                databatch) -> Tensor:
        batch = databatch.batch
        edge_index = databatch.edge_index
        global_mass_info: Dict[str, Tensor] = databatch.global_mass_info

        # Get predefined information
        non_boundary_nodes_mask = ~databatch.boundary_nodes_mask
        inflow_edges_mask = global_mass_info['inflow_edges_mask']
        outflow_edges_mask = global_mass_info['outflow_edges_mask']

        # Get current total water volume (t)
        curr_water_volume = get_orig_water_volume(batch_node_input, self.normalizer, self.is_normalized, non_boundary_nodes_mask)
        non_boundary_batch = batch[non_boundary_nodes_mask]
        total_water_volume = scatter(curr_water_volume, non_boundary_batch, reduce='sum')

        # Get next total water volume (t+1)
        next_water_volume = get_orig_water_volume(batch_node_pred, self.normalizer, self.is_normalized, non_boundary_nodes_mask)
        total_next_water_volume = scatter(next_water_volume, non_boundary_batch, reduce='sum')

        # Get current water inflow (t)
        water_flow = get_orig_water_flow(batch_edge_input, self.normalizer, self.is_normalized)
        curr_inflow = water_flow[inflow_edges_mask]
        inflow_node_idxs = edge_index[0, inflow_edges_mask]
        inflow_batch = batch[inflow_node_idxs]
        total_inflow = scatter(curr_inflow, inflow_batch, reduce='sum')

        # Get current water outflow (t)
        if outflow_edges_mask.sum() == 0:
            total_outflow = torch.zeros_like(total_inflow)
        else:
            curr_outflow = water_flow[outflow_edges_mask]
            outflow_node_idxs = edge_index[1, outflow_edges_mask]
            outflow_batch = batch[outflow_node_idxs]
            total_outflow = scatter(curr_outflow, outflow_batch, reduce='sum')

        # Compute Global Mass Conservation
        delta_v = total_next_water_volume - total_water_volume
        rf_volume = total_rainfall
        inflow_volume = total_inflow * self.delta_t
        outflow_volume = total_outflow * self.delta_t

        global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        if self.mode == 'train':
            global_volume_error = torch.abs(global_volume_error)
            global_loss = global_volume_error.mean()
        else:
            global_loss = global_volume_error.sum()

        return global_loss
