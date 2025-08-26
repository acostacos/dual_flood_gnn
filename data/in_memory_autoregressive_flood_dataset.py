from .autoregressive_flood_dataset import AutoregressiveFloodDataset
from .in_memory_flood_dataset import InMemoryFloodDataset

class InMemoryAutoregressiveFloodDataset(AutoregressiveFloodDataset, InMemoryFloodDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
