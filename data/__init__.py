from typing import Literal

from .auto_flood_event_dataset import AutoregressiveFloodEventDataset
from .flood_event_dataset import FloodEventDataset
from .in_mem_auto_flood_event_dataset import InMemoryAutoregressiveFloodEventDataset
from .in_memory_flood_event_dataset import InMemoryFloodEventDataset

def dataset_factory(storage_mode: Literal['memory', 'disk'], autoregressive: bool, *args, **kwargs) -> FloodEventDataset:
    if autoregressive:
        if storage_mode == 'memory':
            return InMemoryAutoregressiveFloodEventDataset(*args, **kwargs)
        elif storage_mode == 'disk':
            return AutoregressiveFloodEventDataset(*args, **kwargs)

    if storage_mode == 'memory':
        return InMemoryFloodEventDataset(*args, **kwargs)
    elif storage_mode == 'disk':
        return FloodEventDataset(*args, **kwargs)

    raise ValueError(f'Dataset class is not defined.')

__all__ = [
    'AutoregressiveFloodEventDataset',
    'FloodEventDataset',
    'InMemoryAutoregressiveFloodEventDataset',
    'InMemoryFloodEventDataset',
    'dataset_factory',
]
