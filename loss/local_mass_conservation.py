import numpy as np
import torch

from torch import Tensor
from torch_geometric.utils import scatter
from typing import Dict

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition

def get_batch_mask(mask: np.ndarray, num_graphs: int):
    return np.tile(mask, num_graphs)

def get_batch_inflow_outflow(
    edge_index: Tensor,
    face_flow: Tensor,
    batch_outflow_edges_mask: np.ndarray,
    batch_non_boundary_nodes_mask: np.ndarray,
    num_nodes: int,
):
    # Flip outflow edges as these are pointing away from boundary node for message passing.
    # Physics interpretation is that outflow edges point towards the boundary node.
    edge_index[0, batch_outflow_edges_mask], edge_index[1, batch_outflow_edges_mask] =\
        edge_index[1, batch_outflow_edges_mask], edge_index[0, batch_outflow_edges_mask] # Flip edge direction
    face_flow[batch_outflow_edges_mask] *= -1  # Flip flow direction

    face_flow = torch.relu(face_flow) # Negative flow would just be opposite direction of positive flow; Can be ignored
    total_inflow = scatter(face_flow, edge_index[1], reduce='sum', dim_size=num_nodes)
    total_inflow = total_inflow[batch_non_boundary_nodes_mask]
    total_outflow = scatter(face_flow, edge_index[0], reduce='sum', dim_size=num_nodes)
    total_outflow = total_outflow[batch_non_boundary_nodes_mask]

    return total_inflow, total_outflow

def local_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        databatch,
        normalizer: DatasetNormalizer,
        bc_helper: BoundaryCondition,
        is_normalized: bool = True,
        delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    edge_index = databatch.edge_index
    num_nodes = databatch.num_nodes
    num_graphs = databatch.num_graphs
    local_mass_info: Dict[str, Tensor] = databatch.local_mass_info
    device = batch_node_pred.device

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()
    outflow_edges_mask = bc_helper.outflow_edges_mask

    # Get predefined information
    rainfall = local_mass_info['rainfall'].to(device)
    water_volume = local_mass_info['water_volume'].to(device)

    face_flow = local_mass_info['face_flow'].to(device)
    batch_outflow_edges_mask = get_batch_mask(outflow_edges_mask, num_graphs)
    batch_non_boundary_nodes_mask = get_batch_mask(non_boundary_nodes_mask, num_graphs)
    total_inflow, total_outflow = get_batch_inflow_outflow(
        edge_index=edge_index,
        face_flow=face_flow,
        batch_outflow_edges_mask=batch_outflow_edges_mask,
        batch_non_boundary_nodes_mask=batch_non_boundary_nodes_mask,
        num_nodes=num_nodes,
    )

    # Get predictions
    node_pred = batch_node_pred # Normalized predicted water volume (t+1)
    if is_normalized:
        node_pred = normalizer.denormalize('water_volume', node_pred)
    node_pred = torch.relu(node_pred) # Negative water volume would not make sense; Can be ignored
    node_pred = node_pred[batch_non_boundary_nodes_mask]
    next_water_volume = node_pred.squeeze()

    # # Normalized predicted water flow
    # edge_pred = batch_edge_pred[batch == uid] # Normalized predicted water flow
    # if is_normalized:
    #     edge_pred = normalizer.denormalize('face_flow', edge_pred)

    # Local mass conservation equation:
    delta_v = next_water_volume - water_volume
    rf_volume = rainfall
    inflow_volume =  total_inflow * delta_t
    outflow_volume = total_outflow * delta_t

    local_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
    local_volume_error = torch.abs(local_volume_error)
    non_boundary_batch = batch[batch_non_boundary_nodes_mask]
    total_local_volume_error = scatter(local_volume_error, non_boundary_batch, reduce='sum', dim_size=num_graphs)

    local_loss = total_local_volume_error.mean()
    return local_loss

