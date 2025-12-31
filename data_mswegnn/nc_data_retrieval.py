import numpy as np

from utils.file_utils import read_nc_file_as_numpy

def get_face_flow(simulation_path: str, dtype: np.dtype = np.float32):
    property = 'mesh2d_q1'
    data = read_nc_file_as_numpy(simulation_path, property_name=property)
    return data.astype(dtype)

def get_water_depth(simulation_path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    property = 'mesh2d_waterdepth'
    data = read_nc_file_as_numpy(simulation_path, property_name=property)
    return data.astype(dtype)
