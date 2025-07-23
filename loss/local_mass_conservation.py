import torch

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition
from numpy import ndarray
from torch import Tensor
from torch_geometric.utils import scatter
from typing import Dict

from .loss_helper import get_batch_mask, get_orig_water_volume

def get_batch_inflow_outflow(
    edge_index: Tensor,
    face_flow: Tensor,
    batch_non_boundary_nodes_mask: ndarray,
    num_nodes: int,
):
    # Convert edge_index and face_flow to undirected w/ flipped edge features for inflow/outflow calculations
    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], axis=0), torch.cat([col, row], axis=0)
    edge_index = torch.stack([row, col], axis=0)
    face_flow = torch.cat([face_flow, -face_flow], axis=0)

    face_flow = torch.relu(face_flow) # Negative flow would just be opposite direction of positive flow; Can be ignored
    total_inflow = scatter(face_flow, edge_index[1], reduce='sum', dim_size=num_nodes)
    total_inflow = total_inflow[batch_non_boundary_nodes_mask]
    total_outflow = scatter(face_flow, edge_index[0], reduce='sum', dim_size=num_nodes)
    total_outflow = total_outflow[batch_non_boundary_nodes_mask]

    return total_inflow, total_outflow

def local_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        water_volume: Tensor,
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

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()

    # Get predefined information
    rainfall = local_mass_info['rainfall']

    # TODO: Revert once edge prediction is implemented
    face_flow = local_mass_info['face_flow']
    batch_non_boundary_nodes_mask = get_batch_mask(non_boundary_nodes_mask, num_graphs)
    total_inflow, total_outflow = get_batch_inflow_outflow(
        edge_index=edge_index,
        face_flow=face_flow,
        batch_non_boundary_nodes_mask=batch_non_boundary_nodes_mask,
        num_nodes=num_nodes,
    )

    # Get predictions
    # Node prediction = normalized predicted water volume (t+1)
    batch_non_boundary_nodes_mask = get_batch_mask(non_boundary_nodes_mask, num_graphs)
    next_water_volume = get_orig_water_volume(batch_node_pred, normalizer, is_normalized, batch_non_boundary_nodes_mask)

    # TODO: Revert once edge prediction is implemented
    # # Edge prediction = normalized predicted water flow (t+1)
    # if is_normalized:
    #     batch_edge_pred = normalizer.denormalize('face_flow', batch_edge_pred)
    # total_inflow, total_outflow = get_batch_inflow_outflow(
    #     edge_index=edge_index,
    #     face_flow=batch_edge_pred,
    #     batch_non_boundary_nodes_mask=batch_non_boundary_nodes_mask,
    #     num_nodes=num_nodes,
    # )

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

