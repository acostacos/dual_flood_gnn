import numpy as np
import os
import rasterio
import whitebox
import logging

from rasterio import transform

def _get_whitebox_tools() -> whitebox.WhiteboxTools:
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    return wbt

def extract_values_from_dem(dem_path: str, xy_coords: np.ndarray) -> np.ndarray:
    log = logging.getLogger()
    log.setLevel(logging.WARN)  # Suppress rasterio debug info

    with rasterio.open(dem_path) as src:
        band = src.read(1)

        rows, cols = transform.rowcol(src.transform, xy_coords[:, 0], xy_coords[:, 1])

        out_of_bounds = (rows < 0) | (rows >= src.height) | (cols < 0) | (cols >= src.width)
        if np.any(out_of_bounds):
            idx = np.where(out_of_bounds)[0][0]
            x, y = xy_coords[idx]
            raise ValueError(f"Coordinates ({x}, {y}) are out of bounds for the DEM file {dem_path}.")

        value_arr = band[rows, cols]

    log.setLevel(logging.DEBUG)  # Restore logging level

    return value_arr

def get_filled_dem(dem_path: str, output_path: str) -> str:
    assert '.tif' in output_path, "Output file extention must be a .tif file"
    dem_path = os.path.abspath(dem_path)
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path):
        return output_path

    print(f'Creating filled DEM at: {output_path}')
    wbt = _get_whitebox_tools()
    wbt.fill_depressions(dem_path, output_path)
    return output_path

def get_aspect(dem_path: str, output_path: str, xy_coords: np.ndarray) -> np.ndarray:
    assert '.tif' in output_path, "Output file extention must be a .tif file"
    dem_path = os.path.abspath(dem_path)
    output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        print(f'Creating aspect DEM at: {output_path}')
        wbt = _get_whitebox_tools()
        wbt.aspect(dem_path, output_path)
    aspect = extract_values_from_dem(output_path, xy_coords)
    return aspect

def get_curvature(dem_path: str, output_path: str, xy_coords: np.ndarray) -> np.ndarray:
    assert '.tif' in output_path, "Output file extention must be a .tif file"
    dem_path = os.path.abspath(dem_path)
    output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        print(f'Creating curvature DEM at: {output_path}')
        wbt = _get_whitebox_tools()
        # Or do we use wbt.profile_curvature (curvature in the direction of steepest slope)
        wbt.total_curvature(dem_path, output_path)
    curvature = extract_values_from_dem(output_path, xy_coords)
    return curvature

def get_flow_accumulation(dem_path: str, flow_dir_path: str, flow_acc_path: str, xy_coords: np.ndarray) -> np.ndarray:
    assert '.tif' in flow_dir_path and '.tif' in flow_acc_path, "Output file extention must be a .tif file"
    wbt = _get_whitebox_tools()
    dem_path = os.path.abspath(dem_path)
    flow_dir_path = os.path.abspath(flow_dir_path)
    if not os.path.exists(flow_dir_path):
        print(f'Creating flow direction DEM at: {flow_dir_path}')
        wbt.d8_pointer(dem_path, flow_dir_path)
    flow_acc_path = os.path.abspath(flow_acc_path)
    if not os.path.exists(flow_acc_path):
        print(f'Creating flow accumulation DEM at: {flow_acc_path}')
        wbt.d8_flow_accumulation(flow_dir_path, flow_acc_path)
    flow_accum = extract_values_from_dem(flow_acc_path, xy_coords)
    return flow_accum
