import torch
import numpy as np

from torch import Tensor

from data.dataset_normalizer import DatasetNormalizer

def global_mass_conservation_loss(batch_pred: Tensor, databatch, delta_t: int = 30):
    batch = databatch.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    unique_ids = torch.unique(batch)
    global_mass_info = databatch.global_mass_info

    # Values assumed to be the same for all batches
    volume_mean = global_mass_info['volume_mean'][0]
    volume_std = global_mass_info['volume_std'][0]
    inflow_boundary_nodes = databatch.inflow_boundary_nodes[0]
    outflow_boundary_nodes = databatch.outflow_boundary_nodes[0]
    boundary_nodes = np.union1d(inflow_boundary_nodes, outflow_boundary_nodes)

    physics_losses = []
    for uid in unique_ids:
        total_inflow = global_mass_info['total_inflow'][uid]
        total_outflow = global_mass_info['total_outflow'][uid]
        total_rainfall = global_mass_info['total_rainfall'][uid]
        total_next_water_volume = global_mass_info['total_next_water_volume'][uid]

        pred = batch_pred[batch == uid] # Normalized predicted water volume

        # Remove boundary nodes
        non_boundary_nodes_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        non_boundary_nodes_mask[boundary_nodes] = False
        pred = pred[non_boundary_nodes_mask]

        denorm_volume_pred = pred * (volume_std + DatasetNormalizer.EPS) + volume_mean
        total_water_volume = denorm_volume_pred.sum()

        delta_v = total_next_water_volume - total_water_volume
        rf_volume = total_rainfall
        inflow_volume = total_inflow * delta_t
        outflow_volume = total_outflow * delta_t

        global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume
        global_volume_error = global_volume_error.abs()
        physics_losses.append(global_volume_error)

    global_loss = torch.stack(physics_losses).mean()
    return global_loss
