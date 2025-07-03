import torch

from torch import Tensor

from data.dataset_normalizer import DatasetNormalizer
from data.boundary_condition import BoundaryCondition

def global_mass_conservation_loss(
        batch_node_pred: Tensor,
        batch_edge_pred: Tensor,
        databatch,
        normalizer: DatasetNormalizer,
        bc_helper: BoundaryCondition,
        is_normalized: bool = True,
        delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    unique_ids = torch.unique(batch)
    global_mass_info = databatch.global_mass_info

    # Values assumed to be the same for all batches
    non_boundary_nodes_mask = bc_helper.get_non_boundary_nodes_mask()

    physics_losses = []
    for uid in unique_ids:
        total_inflow = global_mass_info['total_inflow'][uid]
        total_outflow = global_mass_info['total_outflow'][uid]
        total_rainfall = global_mass_info['total_rainfall'][uid]
        total_water_volume = global_mass_info['total_water_volume'][uid]

        pred = batch_node_pred[batch == uid] # Normalized predicted water volume (t+1)
        if is_normalized:
            pred = normalizer.denormalize('water_volume', pred)
        pred = torch.relu(pred) # Negative water volume would not make sense; Can be ignored
        pred = pred[non_boundary_nodes_mask]
        total_next_water_volume = pred.sum()

        delta_v = total_next_water_volume - total_water_volume
        rf_volume = total_rainfall
        inflow_volume = total_inflow * delta_t
        outflow_volume = total_outflow * delta_t

        global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        global_volume_error = torch.abs(global_volume_error)
        physics_losses.append(global_volume_error)

    global_loss = torch.stack(physics_losses).mean()
    return global_loss
