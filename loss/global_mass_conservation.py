import torch

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition
from torch import Tensor
from torch_geometric.utils import scatter
from typing import Dict

from .loss_helper import get_batch_mask

def global_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        databatch,
        normalizer: DatasetNormalizer,
        bc_helper: BoundaryCondition,
        is_normalized: bool = True,
        delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    num_graphs = databatch.num_graphs
    global_mass_info: Dict[str, Tensor] = databatch.global_mass_info
    device = batch_node_pred.device

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()

    # Get predefined information
    total_inflow = global_mass_info['total_inflow'].to(device)
    total_outflow = global_mass_info['total_outflow'].to(device)
    total_rainfall = global_mass_info['total_rainfall'].to(device)
    total_water_volume = global_mass_info['total_water_volume'].to(device)

    # Get predictions
    batch_non_boundary_nodes_mask = get_batch_mask(non_boundary_nodes_mask, num_graphs)
    node_pred = batch_node_pred.squeeze() # Normalized predicted water volume (t+1)
    if is_normalized:
        node_pred = normalizer.denormalize('water_volume', node_pred)
    node_pred = torch.relu(node_pred) # Negative water volume would not make sense; Can be ignored
    node_pred = node_pred[batch_non_boundary_nodes_mask]
    non_boundary_batch = batch[batch_non_boundary_nodes_mask]
    total_next_water_volume = scatter(node_pred, non_boundary_batch, reduce='sum')

    delta_v = total_next_water_volume - total_water_volume
    rf_volume = total_rainfall
    inflow_volume = total_inflow * delta_t
    outflow_volume = total_outflow * delta_t

    global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
    global_volume_error = torch.abs(global_volume_error)

    global_loss = global_volume_error.mean()
    return global_loss
