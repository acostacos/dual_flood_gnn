from .auto_mswegnn_flood_dataset import AutomSWEGNNFloodEventDataset
from .mem_mswegnn_flood_dataset import MemmSWEGNNFloodDataset

class MemAutomSWEGNNFloodDataset(AutomSWEGNNFloodEventDataset, MemmSWEGNNFloodDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
