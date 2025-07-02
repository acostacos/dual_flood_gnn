import torch

from torch import Tensor
from torch_geometric.utils import scatter

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition


def local_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        databatch,
        normalizer: DatasetNormalizer,
        bc_helper: BoundaryCondition,
        is_normalized: bool = True,
        delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    unique_ids = torch.unique(batch)
    local_mass_info = databatch.local_mass_info

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()

    physics_losses = []
    for uid in unique_ids:
        rainfall = local_mass_info['rainfall'][batch == uid]
        next_water_volume = local_mass_info['next_water_volume'][batch == uid]

        node_pred = batch_node_pred[batch == uid] # Normalized predicted water volume
        if is_normalized:
            node_pred = normalizer.denormalize('water_volume', node_pred)
        node_pred = node_pred[non_boundary_nodes_mask]
        num_nodes = node_pred.shape[0]

        # Normalized predicted water flow
        edge_pred = batch_edge_pred[batch == uid] # Normalized predicted water flow
        if is_normalized:
            edge_pred = normalizer.denormalize('face_flow', edge_pred)
        edge_pred = torch.relu(edge_pred) # Negative flow would just be opposite direction of positive flow; Can be ignored

        edge_index = databatch.edge_index[batch == uid]
        total_inflow = scatter(edge_pred, edge_index[1], reduce='sum', dim_size=num_nodes)
        total_outflow = scatter(edge_pred, edge_index[0], reduce='sum', dim_size=num_nodes)

        delta_v = next_water_volume - node_pred
        rf_volume = rainfall
        inflow_volume = total_inflow * delta_t # TODO: Change this to be generalized for different time intervals
        outflow_volume = total_outflow * delta_t # TODO: Change this to be generalized for different time intervals

        local_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        total_local_volume_error = torch.abs(local_volume_error).sum()
        physics_losses.append(total_local_volume_error)

    local_loss = torch.stack(physics_losses).mean()
    return local_loss

