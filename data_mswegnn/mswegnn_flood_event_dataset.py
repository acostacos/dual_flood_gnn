import os
import numpy as np

from numpy import ndarray
from data.flood_event_dataset import FloodEventDataset
from data.boundary_condition import BoundaryCondition
from data.shp_data_retrieval import get_cell_elevation, get_cell_position_x, get_cell_position_y,\
    get_cell_position, get_edge_index, get_edge_length, get_edge_slope
from data.dem_data_retrieval import get_filled_dem, get_aspect, get_curvature, get_flow_accumulation
from typing import Callable, List, Literal, Tuple

from .hydrograph_data_retrieval import get_event_timesteps, get_inflow_hydrograph
from .mswegnn_boundary_condition import mSWEGNNBoundaryCondition
from .nc_data_retrieval import get_face_flow, get_water_depth
from .shp_data_retrieval import get_node_types, get_edge_types, get_cell_area, get_face_length

class mSWEGNNFloodEventDataset(FloodEventDataset):
    EVENT_FILE_KEYS = [*FloodEventDataset.EVENT_FILE_KEYS, 'Cells_Shp_Filepath', 'Hydrograph_Filepath']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_from_peak = None  # Currently not implemented in mSWEGNNFloodEventDataset

    def _create_boundary_conditions(self, root_dir: str) -> List[BoundaryCondition]:
        bc_list = []
        for paths, run_id in zip(self.event_file_paths, self.event_run_ids):
            simulation_path, nodes_shp_path = paths[0], paths[1]
            npz_filename = f'boundary_condition_event_{run_id}.npz'
            bc = mSWEGNNBoundaryCondition(root_dir=root_dir,
                                          simulation_file=simulation_path,
                                          nodes_shp_file=nodes_shp_path,
                                          inflow_boundary_nodes=None, # Set within class
                                          outflow_boundary_nodes=None, # Set within class
                                          saved_npz_file=npz_filename)
            bc_list.append(bc)
        return bc_list

    def _set_event_properties(self):
        self._event_peak_idx = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        current_total_ts = 0
        for event_idx in range(len(self.event_run_ids)):
            paths = self._get_event_file_paths(event_idx)

            timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[5]])
            event_ts_interval = int((timesteps[1] - timesteps[0]))
            assert self.timestep_interval % event_ts_interval == 0, f'Event {self.event_run_ids[event_idx]} has a timestep interval of {event_ts_interval} seconds, which is not compatible with the dataset timestep interval of {self.timestep_interval} seconds.'
            self._event_base_timestep_interval.append(event_ts_interval)

            # water_volume = get_water_volume(paths[self.EVENT_FILE_KEYS[0]])
            # total_water_volume = water_volume.sum(axis=1)
            # peak_idx = np.argmax(total_water_volume).item()
            # num_timesteps_after_peak = self.time_from_peak // event_ts_interval if self.time_from_peak is not None else 0
            # assert peak_idx + num_timesteps_after_peak < len(timesteps), "Timesteps after peak exceeds the available timesteps."
            # self._event_peak_idx.append(peak_idx)

            timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
            trim_num_timesteps = len(timesteps)

            event_total_rollout_ts = trim_num_timesteps - self.previous_timesteps - 1  # First timestep starts at self.previous_timesteps; Last timestep is used for labels
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        # assert len(self._event_peak_idx) == len(self.event_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self.event_start_idx) == len(self.event_run_ids), 'Mismatch in number of events and start indices.'

    def _get_event_timesteps(self, event_idx: int) -> ndarray:
        paths = self._get_event_file_paths(event_idx)
        timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[5]])
        timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
        return timesteps

    def _get_static_node_features(self, event_idx: int) -> ndarray:
        def _get_roughness(node_shp_path: str) -> np.ndarray:
            num_nodes = get_node_types(node_shp_path).shape[0]
            ROUGHNESS_VALUE = 0.023  # Manning's n
            roughness = np.full((num_nodes,), ROUGHNESS_VALUE, dtype=np.float32)
            return roughness

        def _get_dem_based_feature(node_shp_path: str,
                                   dem_path: str,
                                   feature_func: Callable,
                                   *output_filenames: Tuple[str]) -> ndarray:
            pos = get_cell_position(node_shp_path)
            dem_folder = os.path.dirname(dem_path)
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            filled_dem_path = os.path.join(dem_folder, f'{dem_filename}_filled.tif')
            filled_dem = get_filled_dem(dem_path, filled_dem_path)

            output_paths = [os.path.join(dem_folder, fn) for fn in output_filenames]
            return feature_func(filled_dem, *output_paths, pos)

        def _get_aspect(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_aspect, f'{dem_filename}_aspect.tif')

        def _get_curvature(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_curvature, f'{dem_filename}_curvature.tif')

        def _get_flow_accumulation(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_flow_accumulation,
                                          f'{dem_filename}_flow_dir.tif', f'{dem_filename}_flow_acc_dem.tif')

        paths = self._get_event_file_paths(event_idx)
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(paths[self.EVENT_FILE_KEYS[4]]),
            "roughness": lambda: _get_roughness(paths[self.EVENT_FILE_KEYS[1]]),
            "elevation": lambda: get_cell_elevation(paths[self.EVENT_FILE_KEYS[1]]),
            "position_x": lambda: get_cell_position_x(paths[self.EVENT_FILE_KEYS[1]]),
            "position_y": lambda: get_cell_position_y(paths[self.EVENT_FILE_KEYS[1]]),
            "aspect": lambda: _get_aspect(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
            "curvature": lambda: _get_curvature(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
            "flow_accumulation": lambda: _get_flow_accumulation(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
        }

        static_features = self._get_features(feature_list=self.STATIC_NODE_FEATURES,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_static_edge_features(self, event_idx: int) -> ndarray:
        def get_relative_position(coord: Literal['x', 'y'], nodes_shp_path: str, edges_shp_path: str) -> ndarray:
            pos_retrieval_func = get_cell_position_x if coord == 'x' else get_cell_position_y
            position = pos_retrieval_func(nodes_shp_path)
            edge_index = get_edge_index(edges_shp_path)
            row, col = edge_index
            relative_pos = position[row] - position[col]
            return relative_pos

        paths = self._get_event_file_paths(event_idx)
        STATIC_EDGE_RETRIEVAL_MAP = {
            "face_length": lambda: get_face_length(paths[self.EVENT_FILE_KEYS[2]]),
            "length": lambda: get_edge_length(paths[self.EVENT_FILE_KEYS[2]]),
            "slope": lambda: get_edge_slope(paths[self.EVENT_FILE_KEYS[2]]),
            "relative_position_x": lambda: get_relative_position('x', paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[2]]),
            "relative_position_y": lambda: get_relative_position('y', paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[2]]),
        }

        static_features = self._get_features(feature_list=self.STATIC_EDGE_FEATURES,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_dynamic_node_features(self, event_idx: int) -> ndarray:
        def _get_rainfall(simulation_path: str, node_shp_path: str):
            water_depth = get_water_depth(simulation_path)
            num_timesteps = water_depth.shape[0]
            node_types = get_node_types(node_shp_path)
            num_nodes = node_types.shape[0]

            # No rainfall in mSWEGNN dataset
            rainfall = np.zeros((num_timesteps, num_nodes), dtype=water_depth.dtype)
            return rainfall

        def _get_inflow_hydrograph(hydrograph_path: str, node_shp_path: str):
            inflow = get_inflow_hydrograph(hydrograph_path)[:, None]
            assert np.all(inflow >= 0), "Inflow hydrograph contains negative values."
            node_types = get_node_types(node_shp_path)
            num_nodes = node_types.shape[0]
            inflow = np.repeat(inflow, repeats=num_nodes, axis=-1)
            return inflow

        def _get_water_volume(simulation_path: str, node_shp_path: str, cells_shp_path: str):
            water_depth = get_water_depth(simulation_path)
            # Create water_depth for ghost nodes as zero
            node_types = get_node_types(node_shp_path)
            num_ghost_nodes = (node_types != 1).sum()
            ghost_nodes_depth = np.zeros((water_depth.shape[0], num_ghost_nodes), dtype=water_depth.dtype)
            water_depth = np.concatenate([water_depth, ghost_nodes_depth], axis=1)

            cell_area = get_cell_area(cells_shp_path)[None, :]
            cell_area = np.repeat(cell_area, repeats=water_depth.shape[0], axis=0)
            water_volume = water_depth * cell_area
            return water_volume

        paths = self._get_event_file_paths(event_idx)
        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "inflow": lambda: self._get_event_dynamic(event_idx, _get_inflow_hydrograph, aggr='mean',
                                                      hydrograph_path=paths[self.EVENT_FILE_KEYS[5]],
                                                      node_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
            "rainfall": lambda: self._get_event_dynamic(event_idx, _get_rainfall, aggr='sum',
                                                        simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                        node_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
            "water_volume": lambda: self._get_event_dynamic(event_idx, _get_water_volume, aggr='mean',
                                                            simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                            node_shp_path=paths[self.EVENT_FILE_KEYS[1]],
                                                            cells_shp_path=paths[self.EVENT_FILE_KEYS[4]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self, event_idx: int) -> ndarray:
        def _get_face_flow(simulation_path: str, hydrograph_path: str, edges_shp_path: str):
            face_flow = get_face_flow(simulation_path)

            # Overwrite boundary edge flows with inflow hydrograph
            # Found that the simulation output differs for the first timestep
            inflow = get_inflow_hydrograph(hydrograph_path)[:, None]
            edge_types = get_edge_types(edges_shp_path)
            boundary_edge_indices = np.where(edge_types == 2)[0]
            face_flow[:, boundary_edge_indices] = inflow

            return face_flow

        paths = self._get_event_file_paths(event_idx)
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "face_flow": lambda: self._get_event_dynamic(event_idx, _get_face_flow, aggr='mean',
                                                         simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                         hydrograph_path=paths[self.EVENT_FILE_KEYS[5]],
                                                         edges_shp_path=paths[self.EVENT_FILE_KEYS[2]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features
