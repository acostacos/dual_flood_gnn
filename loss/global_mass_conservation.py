import torch

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition
from torch import Tensor
from torch_geometric.utils import scatter
from typing import Dict

from .loss_helper import get_batch_mask, get_orig_water_volume

def global_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        total_water_volume: Tensor,
        databatch,
        normalizer: DatasetNormalizer,
        bc_helper: BoundaryCondition,
        is_normalized: bool = True,
        delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    edge_index = databatch.edge_index
    num_graphs = databatch.num_graphs
    global_mass_info: Dict[str, Tensor] = databatch.global_mass_info

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = ~bc_helper.boundary_nodes_mask
    outflow_edges_mask = bc_helper.outflow_edges_mask

    # Get predefined information
    total_inflow = global_mass_info['total_inflow']
    total_rainfall = global_mass_info['total_rainfall']

    # TODO: Revert once edge prediction is implemented
    batch_outflow_edges_mask = get_batch_mask(outflow_edges_mask, num_graphs)
    face_flow = global_mass_info['face_flow']
    outflow = face_flow[batch_outflow_edges_mask].squeeze()
    outflow_node_idxs = edge_index[1, batch_outflow_edges_mask]
    outflow_batch = batch[outflow_node_idxs]
    total_outflow = scatter(outflow, outflow_batch, reduce='sum')

    # Get predictions
    # Node prediction = normalized predicted water volume (t+1)
    batch_non_boundary_nodes_mask = get_batch_mask(non_boundary_nodes_mask, num_graphs)
    next_water_volume = get_orig_water_volume(batch_node_pred, normalizer, is_normalized, batch_non_boundary_nodes_mask)
    non_boundary_batch = batch[batch_non_boundary_nodes_mask]
    total_next_water_volume = scatter(next_water_volume, non_boundary_batch, reduce='sum')

    # TODO: Revert once edge prediction is implemented
    # # Edge prediction = normalized predicted water flow (t+1)
    # batch_outflow_edges_mask = get_batch_mask(outflow_edges_mask, num_graphs)
    # if is_normalized:
    #     batch_edge_pred = normalizer.denormalize('face_flow', batch_edge_pred)
    # outflow = batch_edge_pred[batch_outflow_edges_mask].squeeze()
    # outflow_node_idxs = edge_index[1, batch_outflow_edges_mask]
    # outflow_batch = batch[outflow_node_idxs]
    # total_outflow = scatter(outflow, outflow_batch, reduce='sum')

    delta_v = total_next_water_volume - total_water_volume
    rf_volume = total_rainfall
    inflow_volume = total_inflow * delta_t
    outflow_volume = total_outflow * delta_t

    global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
    global_volume_error = torch.abs(global_volume_error)

    global_loss = global_volume_error.mean()
    return global_loss
