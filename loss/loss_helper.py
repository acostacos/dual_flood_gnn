import numpy as np

def get_batch_mask(mask: np.ndarray, num_graphs: int):
    return np.tile(mask, num_graphs)
