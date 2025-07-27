import torch

from data.dataset_normalizer import DatasetNormalizer

def get_orig_water_volume(water_volume: torch.Tensor,
                          normalizer: DatasetNormalizer,
                          is_normalized: bool,
                          non_boundary_nodes_mask: bool):
    if is_normalized:
        water_volume = normalizer.denormalize('water_volume', water_volume)
    water_volume = torch.relu(water_volume) # Water volume must be non-negative
    water_volume = water_volume[non_boundary_nodes_mask]
    return water_volume.squeeze()

def get_orig_water_flow(water_flow: torch.Tensor,
                        normalizer: DatasetNormalizer,
                        is_normalized: bool):
    if is_normalized:
        water_flow = normalizer.denormalize('face_flow', water_flow)
    water_flow = water_flow.squeeze()
    return water_flow
