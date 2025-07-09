import torch
import numpy as np

from data.dataset_normalizer import DatasetNormalizer

def get_batch_mask(mask: np.ndarray, num_graphs: int):
    return np.tile(mask, num_graphs)

def get_orig_water_volume(water_volume: torch.Tensor,
                          normalizer: DatasetNormalizer,
                          is_normalized: bool,
                          non_boundary_nodes_mask: bool):
    if is_normalized:
        water_volume = normalizer.denormalize('water_volume', water_volume)
    water_volume = torch.relu(water_volume) # Water volume must be non-negative
    water_volume = water_volume[non_boundary_nodes_mask]
    return water_volume
