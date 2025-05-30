import numpy as np
import geopandas as gpd

def read_shp_file_as_numpy(filepath: str, columns: str | list) -> np.ndarray:
    file = gpd.read_file(filepath)
    np_data = file[columns].to_numpy()
    return np_data

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
