from typing import Literal

from .autoregressive_flood_dataset import AutoregressiveFloodDataset
from .flood_event_dataset import FloodEventDataset
from .in_memory_autoregressive_flood_dataset import InMemoryAutoregressiveFloodDataset
from .in_memory_flood_dataset import InMemoryFloodDataset

def dataset_factory(storage_mode: Literal['memory', 'disk'], autoregressive: bool, *args, **kwargs) -> FloodEventDataset:
    if autoregressive:
        if storage_mode == 'memory':
            return InMemoryAutoregressiveFloodDataset(*args, **kwargs)
        elif storage_mode == 'disk':
            return AutoregressiveFloodDataset(*args, **kwargs)

    if storage_mode == 'memory':
        return InMemoryFloodDataset(*args, **kwargs)
    elif storage_mode == 'disk':
        return FloodEventDataset(*args, **kwargs)

    raise ValueError(f'Dataset class is not defined.')

__all__ = [
    'AutoregressiveFloodDataset',
    'FloodEventDataset',
    'InMemoryAutoregressiveFloodDataset',
    'InMemoryFloodDataset',
    'dataset_factory',
]
