import numpy as np

from utils.file_utils import read_shp_file_as_numpy

def get_node_types(filepath: str, dtype: np.dtype = np.int32) -> np.ndarray:
    columns = 'node_type'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_types(filepath: str, dtype: np.dtype = np.int32) -> np.ndarray:
    columns = 'edge_type'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_cell_area(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'area_m2'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_face_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'fc_length'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)
