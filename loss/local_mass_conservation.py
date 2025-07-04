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
    device = batch_node_pred.device

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()
    outflow_edges_mask = bc_helper.outflow_edges_mask

    physics_losses = []
    for uid in unique_ids:
        rainfall = local_mass_info['rainfall'][uid]
        rainfall = torch.from_numpy(rainfall).to(device)
        water_volume = local_mass_info['water_volume'][uid]
        water_volume = torch.from_numpy(water_volume).to(device)
        face_flow = local_mass_info['face_flow'][uid]
        face_flow = torch.from_numpy(face_flow).to(device)

        node_pred = batch_node_pred[batch == uid] # Normalized predicted water volume (t+1)
        orig_num_nodes = node_pred.shape[0]
        if is_normalized:
            node_pred = normalizer.denormalize('water_volume', node_pred)
        node_pred = torch.relu(node_pred) # Negative water volume would not make sense; Can be ignored
        node_pred = node_pred[non_boundary_nodes_mask]
        next_water_volume = node_pred.squeeze()

        # # Normalized predicted water flow
        # edge_pred = batch_edge_pred[batch == uid] # Normalized predicted water flow
        # if is_normalized:
        #     edge_pred = normalizer.denormalize('face_flow', edge_pred)

        node_idxs = (batch == uid).nonzero().squeeze()
        edge_index_mask = torch.all(torch.isin(databatch.edge_index, node_idxs), dim=0)
        edge_index = databatch.edge_index[:, edge_index_mask]

        # Flip outflow edges as these are pointing away from boundary node for message passing. Physics interpretation is that outflow edges point towards the boundary node.
        edge_index[0, outflow_edges_mask], edge_index[1, outflow_edges_mask] = edge_index[1, outflow_edges_mask], edge_index[0, outflow_edges_mask] # Flip edge direction
        face_flow[outflow_edges_mask] *= -1  # Flip flow direction

        face_flow = torch.relu(face_flow) # Negative flow would just be opposite direction of positive flow; Can be ignored
        total_inflow = scatter(face_flow, edge_index[1], reduce='sum', dim_size=orig_num_nodes)
        total_inflow = total_inflow[non_boundary_nodes_mask]
        total_outflow = scatter(face_flow, edge_index[0], reduce='sum', dim_size=orig_num_nodes)
        total_outflow = total_outflow[non_boundary_nodes_mask]

        delta_v = next_water_volume - water_volume
        rf_volume = rainfall
        inflow_volume = total_inflow * delta_t
        outflow_volume = total_outflow * delta_t

        local_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        total_local_volume_error = torch.abs(local_volume_error).sum()
        physics_losses.append(total_local_volume_error)

    local_loss = torch.stack(physics_losses).mean()
    return local_loss

