import numpy as np

from utils.file_utils import read_shp_file_as_numpy

def get_cell_position_x(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'X'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_cell_position_y(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'Y'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_cell_position(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = ['X', 'Y']
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_index(filepath: str) -> np.ndarray:
    columns = ['from_node', 'to_node']
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return data.astype(np.int64).transpose()

def get_cell_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'Elevation1'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'length'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_slope(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'slope'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)
