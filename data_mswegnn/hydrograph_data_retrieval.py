import numpy as np

def get_event_timesteps(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    data = np.loadtxt(filepath)
    return data[:, 0].astype(dtype)

def get_inflow_hydrograph(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    data = np.loadtxt(filepath)
    hydrograph = data[:, 1]
    return hydrograph.astype(dtype)
