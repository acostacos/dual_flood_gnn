import numpy as np
import h5py

from datetime import datetime
from typing import List, Any

def read_hdf_file_as_numpy(filepath: str, property_path: str, separator: str = '.') -> np.ndarray:
    with h5py.File(filepath, 'r') as hec:
        data = get_property_from_path(hec, property_path, separator)
        np_data = np.array(data)
    return np_data

def get_property_from_path(dict: dict, dict_path: str, separator: str = '.') -> Any:
    keys = dict_path.split(sep=separator)
    d = dict
    for key in keys:
        if key in d:
            d = d[key]
        else:
            raise KeyError(f'Key {key} not found in dictionary for path {dict_path}')
    return d

def get_event_timesteps(filepath: str) -> List[datetime]:
    property_path = 'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Time Date Stamp'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)

    def format(x: np.bytes_) -> datetime:
        TIMESTAMP_FORMAT = '%d%b%Y %H:%M:%S'
        time_str = x.decode('UTF-8')
        time_stamp = datetime.strptime(time_str, TIMESTAMP_FORMAT)
        return time_stamp

    vec_format = np.vectorize(format)
    time_series = vec_format(data)
    return list(time_series)

def get_cell_area(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Surface Area'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_roughness(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f"Geometry.2D Flow Areas.{perimeter_name}.Cells Center Manning's n"
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_rainfall(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Cumulative Precipitation Depth'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Convert cumulative rainfall to interval rainfall
    first_ts_rainfall = np.zeros((1, data.shape[1]), dtype=np.float32)
    intervals = np.diff(data, axis=0)
    data = np.concatenate((first_ts_rainfall, intervals), axis=0)

    return data

def get_water_level(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Water Surface'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_edge_direction_x(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get x direction property
    data = np.squeeze(np.take(data, indices=[0], axis=1))

    return data

def get_edge_direction_y(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get y direction property
    data = np.squeeze(np.take(data, indices=[1], axis=1))

    return data

def get_face_length(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get face length property
    data = np.squeeze(np.take(data, indices=[2], axis=1))

    return data

def get_velocity(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Face Velocity'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)
