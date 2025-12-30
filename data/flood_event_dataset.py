import os
import torch
import numpy as np
import pandas as pd

from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Callable, Tuple, List, Literal, Dict, Optional, Union
from utils.logger import Logger
from utils.file_utils import read_yaml_file, save_to_yaml_file

from .dem_data_retrieval import get_filled_dem, get_aspect, get_curvature, get_flow_accumulation
from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_roughness,\
    get_cumulative_rainfall, get_water_level, get_water_volume, get_edge_direction_x,\
    get_edge_direction_y, get_face_length, get_velocity, get_face_flow
from .shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length,\
    get_edge_slope, get_cell_position_x, get_cell_position_y, get_cell_position
from .boundary_condition import BoundaryCondition
from .dataset_normalizer import DatasetNormalizer

class FloodEventDataset(Dataset):
    EVENT_FILE_KEYS = ['Simulation_Filepath', 'Nodes_Shp_Filepath', 'Edges_Shp_Filepath', 'DEM_Filepath']
    STATIC_NODE_FEATURES = ['position_x', 'position_y', 'area', 'roughness', 'elevation', 'aspect', 'curvature', 'flow_accumulation']
    DYNAMIC_NODE_FEATURES = ['inflow', 'rainfall', 'water_volume'] # Not included: 'water_depth'
    STATIC_EDGE_FEATURES = ['relative_position_x', 'relative_position_y', 'face_length', 'length', 'slope']
    DYNAMIC_EDGE_FEATURES = ['face_flow'] # Not included: 'velocity'
    NODE_TARGET_FEATURE = 'water_volume'
    EDGE_TARGET_FEATURE = 'face_flow'

    def __init__(self,
                 mode: Literal['train', 'test'],
                 root_dir: str,
                 dataset_summary_file: str,
                 event_stats_file: str = 'event_stats.yaml',
                 features_stats_file: str = 'features_stats.yaml',
                 previous_timesteps: int = 2,
                 normalize: bool = True,
                 timestep_interval: int = 30, # in seconds
                 spin_up_time: Union[int, Dict[str, int]] = None,
                 time_from_peak: Optional[int] = None,
                 inflow_boundary_nodes: List[int] = [],
                 outflow_boundary_nodes: List[int] = [],
                 with_global_mass_loss: bool = True,
                 with_local_mass_loss: bool = True,
                 debug: bool = False,
                 logger: Optional[Logger] = None,
                 force_reload: bool = False):
        assert mode in ['train', 'test'], f'Invalid mode: {mode}. Must be "train" or "test".'

        self.log_func = print
        if logger is not None and hasattr(logger, 'log'):
            self.log_func = logger.log

        # File paths
        self.event_file_paths, self.event_run_ids = self._get_events_from_summary(root_dir, dataset_summary_file)
        self.event_stats_file = event_stats_file
        self.features_stats_file = features_stats_file

        # Dataset configurations
        self.mode = mode
        self.previous_timesteps = previous_timesteps
        self.is_normalized = normalize
        self.timestep_interval = timestep_interval
        self.spin_up_time = spin_up_time
        self.time_from_peak = time_from_peak
        self.inflow_boundary_nodes = inflow_boundary_nodes
        self.outflow_boundary_nodes = outflow_boundary_nodes
        self.with_global_mass_loss = with_global_mass_loss
        self.with_local_mass_loss = with_local_mass_loss

        # Dataset variables
        self.num_static_node_features = len(self.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(self.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(self.DYNAMIC_EDGE_FEATURES)
        self.num_processed_files_per_event = 3
        event_stats = self._load_event_stats(root_dir, event_stats_file)
        self.event_start_idx, self.total_rollout_timesteps, processed_event_info = event_stats
        self._event_peak_idx = None
        self._event_base_timestep_interval = None

        # Helper classes
        self.normalizer = DatasetNormalizer(mode, root_dir, features_stats_file)
        self.boundary_conditions = self._create_boundary_conditions(root_dir)

        force_reload = self._is_previous_config_different(processed_event_info) or force_reload
        super().__init__(root_dir, transform=None, pre_transform=None, pre_filter=None, log=debug, force_reload=force_reload)

    @property
    def raw_file_names(self):
        names = []
        for event_paths in self.event_file_paths:
            assert len(event_paths) == len(self.EVENT_FILE_KEYS), f'Each event must have {len(self.EVENT_FILE_KEYS)} files. Found {len(event_paths)} files.'
            names.extend(event_paths)
        return names

    @property
    def processed_file_names(self):
        event_files = []
        for run_id in self.event_run_ids:
            event_files.extend([
                f'static_values_event_{run_id}.npz',
                f'dynamic_values_event_{run_id}.npz',
                f'boundary_condition_event_{run_id}.npz',
            ])
        return [
            self.event_stats_file,
            self.features_stats_file,
            *event_files,
        ]

    def download(self):
        # Data must be downloaded manually and placed in the raw dir
        pass

    def process(self):
        self.log_func('Processing Flood Event Dataset...')

        self._set_event_properties()

        feature_types = ['static_nodes', 'dynamic_nodes', 'static_edges', 'dynamic_edges']
        event_data = {}                                           # Holds feature data for each event
        feature_norm_data = {k: None for k in feature_types}      # Flattened feature data for normalization
        for event_idx, run_id in enumerate(self.event_run_ids):
            # Retrieve event features
            edge_index = self._get_edge_index(event_idx)
            event_timesteps = self._get_event_timesteps(event_idx)
            static_nodes = self._get_static_node_features(event_idx)
            dynamic_nodes = self._get_dynamic_node_features(event_idx)
            static_edges = self._get_static_edge_features(event_idx)
            dynamic_edges = self._get_dynamic_edge_features(event_idx)

            # Apply boundary conditions
            event_bc = self.boundary_conditions[event_idx]
            event_bc.create(edge_index, dynamic_edges)
            static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = event_bc.remove(
                static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index,
            )
            static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = event_bc.apply(
                static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index,
            )

            # Get physics-informed loss features
            node_rainfall_per_ts = self._get_physics_info(dynamic_nodes)

            event_data[event_idx] = {
                'edge_index': edge_index,
                'event_timesteps': event_timesteps,
                'static_nodes': static_nodes,
                'dynamic_nodes': dynamic_nodes,
                'static_edges': static_edges,
                'dynamic_edges': dynamic_edges,
                'node_rainfall_per_ts': node_rainfall_per_ts,
            }

            if self.mode == 'train' and self.is_normalized:
                # Collect data for normalization
                feature_type_values = [static_nodes, dynamic_nodes, static_edges, dynamic_edges]
                for feature_type, values in zip(feature_types, feature_type_values):
                    flattened_values = values.reshape(-1, values.shape[-1]).transpose() # Shape: (num_features, num_samples)
                    if feature_norm_data[feature_type] is None:
                        feature_norm_data[feature_type] = flattened_values
                    else:
                        feature_norm_data[feature_type] = np.concatenate([feature_norm_data[feature_type], flattened_values], axis=1)

        if self.is_normalized:
            feature_type_list = [
                self.STATIC_NODE_FEATURES, self.DYNAMIC_NODE_FEATURES,
                self.STATIC_EDGE_FEATURES, self.DYNAMIC_EDGE_FEATURES,
            ]

            if self.mode == 'train':
                # Update normalization stats
                for feature_type, feature_list in zip(feature_types, feature_type_list):
                    for i, feature_name in enumerate(feature_list):
                        all_values = feature_norm_data[feature_type][i]
                        self.normalizer.update_stats(feature_name, all_values)

            # Normalize features
            for event_idx in range(len(self.event_run_ids)):
                for feature_type, feature_list in zip(feature_types, feature_type_list):
                    event_data[event_idx][feature_type] = \
                        self.normalizer.normalize_feature_vector(feature_list, event_data[event_idx][feature_type])

        # Save processed data
        for event_idx, run_id in enumerate(self.event_run_ids):
            start_idx = 2 + event_idx * self.num_processed_files_per_event
            end_idx = start_idx + self.num_processed_files_per_event
            static_npz_path, dynamic_npz_path, boundary_npz_path = self.processed_paths[start_idx:end_idx]

            np.savez(static_npz_path,
                     edge_index=event_data[event_idx]['edge_index'],
                     static_nodes=event_data[event_idx]['static_nodes'],
                     static_edges=event_data[event_idx]['static_edges'])

            np.savez(dynamic_npz_path,
                     event_timesteps=event_data[event_idx]['event_timesteps'],
                     dynamic_nodes=event_data[event_idx]['dynamic_nodes'],
                     dynamic_edges=event_data[event_idx]['dynamic_edges'],
                     node_rainfall_per_ts=event_data[event_idx]['node_rainfall_per_ts'])

            self.boundary_conditions[event_idx].save_data()

            self.log_func(f'Saved processed info for event {run_id} to {static_npz_path} (static), {dynamic_npz_path} (dynamic) and {boundary_npz_path} (boundary conditions)')

        self._save_event_stats()
        self.log_func(f'Saved event stats to {self.processed_paths[0]}')
        if self.mode == 'train':
            self.normalizer.save_feature_stats()
            self.log_func(f'Saved feature stats to {self.processed_paths[1]}')

    def len(self):
        return self.total_rollout_timesteps

    def get(self, idx):
        # Find the event this index belongs to using the start indices
        if idx < 0 or idx >= self.total_rollout_timesteps:
            raise IndexError(f'Index {idx} out of bounds for dataset with {self.total_rollout_timesteps} timesteps.')
        start_idx = 0
        for si in self.event_start_idx:
            if idx < si:
                break
            start_idx = si
        event_idx = self.event_start_idx.index(start_idx)

        start_idx = 2 + event_idx * self.num_processed_files_per_event
        end_idx = start_idx + self.num_processed_files_per_event
        static_npz_path, dynamic_npz_path, _ = self.processed_paths[start_idx:end_idx]

        # Load static data
        static_values = np.load(static_npz_path)
        edge_index: ndarray = static_values['edge_index']
        static_nodes: ndarray = static_values['static_nodes']
        static_edges: ndarray = static_values['static_edges']

        edge_index = torch.from_numpy(edge_index)
        event_bc = self.boundary_conditions[event_idx]
        boundary_nodes_mask = torch.from_numpy(event_bc.boundary_nodes_mask)
        boundary_edges_mask = torch.from_numpy(event_bc.boundary_edges_mask)

        # Load dynamic data
        dynamic_values = np.load(dynamic_npz_path, allow_pickle=True)
        event_timesteps: ndarray = dynamic_values['event_timesteps']
        dynamic_nodes: ndarray = dynamic_values['dynamic_nodes']
        dynamic_edges: ndarray = dynamic_values['dynamic_edges']

        # Create Data object for timestep
        within_event_idx = idx - start_idx + self.previous_timesteps # First timestep starts at self.previous_timesteps
        timestep = event_timesteps[within_event_idx]
        node_features = self._get_node_timestep_data(static_nodes, dynamic_nodes, within_event_idx)
        edge_features = self._get_edge_timestep_data(static_edges, dynamic_edges, within_event_idx)
        label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, within_event_idx)

        # Get physics-informed loss information
        global_mass_info = None
        local_mass_info = None
        if self.with_global_mass_loss or self.with_local_mass_loss:
            node_rainfall_per_ts: ndarray = dynamic_values['node_rainfall_per_ts']
            if self.with_global_mass_loss:
                global_mass_info = self._get_global_mass_info_for_timestep(node_rainfall_per_ts, event_idx, within_event_idx)

            if self.with_local_mass_loss:
                local_mass_info = self._get_local_mass_info_for_timestep(node_rainfall_per_ts, event_idx, within_event_idx)

        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_features,
                    y=label_nodes,
                    y_edge=label_edges,
                    timestep=timestep,
                    boundary_nodes_mask=boundary_nodes_mask,
                    boundary_edges_mask=boundary_edges_mask,
                    global_mass_info=global_mass_info,
                    local_mass_info=local_mass_info)

        return data

    def _load_event_stats(self, root_dir: str, event_stats_file: str) -> Tuple[List[int], int, Dict]:
        event_stats_path = os.path.join(root_dir, 'processed', event_stats_file)
        if not os.path.exists(event_stats_path):
            return [], 0, {}

        event_stats = read_yaml_file(event_stats_path)
        event_start_idx = event_stats['event_start_idx']
        total_rollout_timesteps = event_stats['total_rollout_timesteps']
        processed_event_info = {
            'timestep_interval': event_stats['timestep_interval'],
            'previous_timesteps': event_stats['previous_timesteps'],
            'normalize': event_stats['normalize'],
            'spin_up_time': event_stats['spin_up_time'],
            'time_from_peak': event_stats['time_from_peak'],
            'inflow_boundary_nodes': event_stats['inflow_boundary_nodes'],
            'outflow_boundary_nodes': event_stats['outflow_boundary_nodes'],
        }
        return event_start_idx, total_rollout_timesteps, processed_event_info

    def _save_event_stats(self):
        event_stats = {
            'event_start_idx': self.event_start_idx,
            'total_rollout_timesteps': self.total_rollout_timesteps,
            'timestep_interval': self.timestep_interval,
            'previous_timesteps': self.previous_timesteps,
            'normalize': self.is_normalized,
            'spin_up_time': self.spin_up_time,
            'time_from_peak': self.time_from_peak,
            'inflow_boundary_nodes': self.inflow_boundary_nodes,
            'outflow_boundary_nodes': self.outflow_boundary_nodes,
        }
        save_to_yaml_file(self.processed_paths[0], event_stats)

    def _get_events_from_summary(self, root_dir: str, dataset_summary_file: str) -> Tuple[List[str], List[str]]:
        dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'
        assert all(key in summary_df.columns for key in self.EVENT_FILE_KEYS), f'Summary file must contain the following columns: {self.EVENT_FILE_KEYS}'

        run_ids = []
        file_paths = []
        for _, row in summary_df.iterrows():
            run_id = row['Run_ID']
            paths = row[self.EVENT_FILE_KEYS].tolist()

            assert run_id not in run_ids, f'Duplicate Run_ID found: {run_id}'
            for p in paths:
                full_path = os.path.join(root_dir, 'raw', p)
                assert os.path.exists(full_path), f'File not found: {p}'

            run_ids.append(run_id)
            file_paths.append(paths)

        return file_paths, run_ids
    
    def _create_boundary_conditions(self, root_dir: str) -> List[BoundaryCondition]:
        bc_list = []
        for paths, run_id in zip(self.event_file_paths, self.event_run_ids):
            simulation_path = paths[0]
            npz_filename = f'boundary_condition_event_{run_id}.npz'
            bc = BoundaryCondition(root_dir=root_dir,
                                   simulation_file=simulation_path,
                                   inflow_boundary_nodes=self.inflow_boundary_nodes,
                                   outflow_boundary_nodes=self.outflow_boundary_nodes,
                                   saved_npz_file=npz_filename)
            bc_list.append(bc)
        return bc_list

    def _is_previous_config_different(self, processed_event_info: Dict) -> bool:
        if processed_event_info is None or len(processed_event_info) == 0:
            self.log_func('No previous event stats found. Processing dataset.')
            return True
        if processed_event_info['timestep_interval'] != self.timestep_interval:
            self.log_func(f'Previous timestep interval {processed_event_info["timestep_interval"]} differs from current {self.timestep_interval}. Reprocessing dataset.')
            return True
        if processed_event_info['previous_timesteps'] != self.previous_timesteps:
            self.log_func(f'Previous previous_timesteps {processed_event_info["previous_timesteps"]} differs from current {self.previous_timesteps}. Reprocessing dataset.')
            return True
        if processed_event_info['normalize'] != self.is_normalized:
            self.log_func(f'Previous normalize {processed_event_info["normalize"]} differs from current {self.is_normalized}. Reprocessing dataset.')
            return True
        if processed_event_info['spin_up_time'] != self.spin_up_time:
            self.log_func(f'Previous spin_up_time {processed_event_info["spin_up_time"]} differs from current {self.spin_up_time}. Reprocessing dataset.')
            return True
        if processed_event_info['time_from_peak'] != self.time_from_peak:
            self.log_func(f'Previous time_from_peak {processed_event_info["time_from_peak"]} differs from current {self.time_from_peak}. Reprocessing dataset.')
            return True
        if set(processed_event_info['inflow_boundary_nodes']) != set(self.inflow_boundary_nodes):
            self.log_func(f'Previous inflow_boundary_nodes {processed_event_info["inflow_boundary_nodes"]} differs from current {self.inflow_boundary_nodes}. Reprocessing dataset.')
            return True
        if set(processed_event_info['outflow_boundary_nodes']) != set(self.outflow_boundary_nodes):
            self.log_func(f'Previous outflow_boundary_nodes {processed_event_info["outflow_boundary_nodes"]} differs from current {self.outflow_boundary_nodes}. Reprocessing dataset.')
            return True
        return False

    # =========== process() methods ===========

    def _set_event_properties(self):
        self._event_peak_idx = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        current_total_ts = 0
        for event_idx in range(len(self.event_run_ids)):
            paths = self._get_event_file_paths(event_idx)

            timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[0]])
            event_ts_interval = int((timesteps[1] - timesteps[0]).total_seconds())
            assert self.timestep_interval % event_ts_interval == 0, f'Event {self.event_run_ids[event_idx]} has a timestep interval of {event_ts_interval} seconds, which is not compatible with the dataset timestep interval of {self.timestep_interval} seconds.'
            self._event_base_timestep_interval.append(event_ts_interval)

            water_volume = get_water_volume(paths[self.EVENT_FILE_KEYS[0]])
            total_water_volume = water_volume.sum(axis=1)
            peak_idx = np.argmax(total_water_volume).item()
            num_timesteps_after_peak = self.time_from_peak // event_ts_interval if self.time_from_peak is not None else 0
            assert peak_idx + num_timesteps_after_peak < len(timesteps), "Timesteps after peak exceeds the available timesteps."
            self._event_peak_idx.append(peak_idx)

            timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
            trim_num_timesteps = len(timesteps)

            event_total_rollout_ts = trim_num_timesteps - self.previous_timesteps - 1  # First timestep starts at self.previous_timesteps; Last timestep is used for labels
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        assert len(self._event_peak_idx) == len(self.event_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self.event_start_idx) == len(self.event_run_ids), 'Mismatch in number of events and start indices.'

    def _get_event_file_paths(self, event_idx: int) -> Dict[str, str]:
        event_file_index = event_idx * len(self.EVENT_FILE_KEYS)
        event_file_paths = {}
        for i, file_key in enumerate(self.EVENT_FILE_KEYS):
            event_file_paths[file_key] = self.raw_paths[event_file_index + i]
        return event_file_paths

    def _get_edge_index(self, event_idx: int) -> ndarray:
        paths = self._get_event_file_paths(event_idx)
        edge_index = get_edge_index(paths[self.EVENT_FILE_KEYS[2]])
        return edge_index

    def _get_event_timesteps(self, event_idx: int) -> ndarray:
        paths = self._get_event_file_paths(event_idx)
        timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[0]])
        timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
        return timesteps

    def _get_static_node_features(self, event_idx: int) -> ndarray:
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
            "area": lambda: get_cell_area(paths[self.EVENT_FILE_KEYS[0]]),
            "roughness": lambda: get_roughness(paths[self.EVENT_FILE_KEYS[0]]),
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
            "direction_x": lambda: get_edge_direction_x(paths[self.EVENT_FILE_KEYS[0]]),
            "direction_y": lambda: get_edge_direction_y(paths[self.EVENT_FILE_KEYS[0]]),
            "face_length": lambda: get_face_length(paths[self.EVENT_FILE_KEYS[0]]),
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
        def get_interval_rainfall(hec_ras_path: str):
            """Get interval rainfall from cumulative rainfall"""
            cumulative_rainfall = get_cumulative_rainfall(hec_ras_path)
            last_ts_rainfall = np.zeros((1, cumulative_rainfall.shape[1]), dtype=cumulative_rainfall.dtype)
            intervals = np.diff(cumulative_rainfall, axis=0)
            interval_rainfall = np.concatenate((intervals, last_ts_rainfall), axis=0)
            return interval_rainfall

        def get_inflow_hydrograph(hec_ras_path: str, edge_shp_path: str, inflow_boundary_nodes: List[int]):
            """Get inflow at boundary nodes"""
            edge_index = get_edge_index(edge_shp_path)
            num_nodes = edge_index.max() + 1
            inflow_to_boundary_mask = np.isin(edge_index[1], inflow_boundary_nodes)
            inflow_edges_mask = np.any(np.isin(edge_index, inflow_boundary_nodes), axis=0)

            face_flow = get_face_flow(hec_ras_path)
            if np.any(inflow_to_boundary_mask):
                # Flip the dynamic edge features accordingly
                face_flow[:, inflow_to_boundary_mask] *= -1
            inflow = face_flow[:, inflow_edges_mask].sum(axis=1)[:, None]
            inflow = np.repeat(inflow, num_nodes, axis=-1)
            return inflow

        def get_water_depth(hec_ras_path: str, nodes_shp_path: str):
            """Get water depth from water level and elevation"""
            water_level = get_water_level(hec_ras_path)
            elevation = get_cell_elevation(nodes_shp_path)[None, :]
            water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
            return water_depth

        def get_clipped_water_volume(hec_ras_path: str):
            """Remove exterme values in water volume"""
            CLIP_VOLUME = 100000  # in cubic meters
            water_volume = get_water_volume(hec_ras_path)
            water_volume = np.clip(water_volume, a_min=0, a_max=CLIP_VOLUME)
            return water_volume

        paths = self._get_event_file_paths(event_idx)
        event_bc = self.boundary_conditions[event_idx]
        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "inflow": lambda: self._get_event_dynamic(event_idx, get_inflow_hydrograph, aggr='mean',
                                                      hec_ras_path=paths[self.EVENT_FILE_KEYS[0]],
                                                      edge_shp_path=paths[self.EVENT_FILE_KEYS[2]],
                                                      inflow_boundary_nodes=event_bc.init_inflow_boundary_nodes),
            "rainfall": lambda: self._get_event_dynamic(event_idx, get_interval_rainfall, aggr='sum',
                                                        hec_ras_path=paths[self.EVENT_FILE_KEYS[0]]),
            "water_depth": lambda: self._get_event_dynamic(event_idx, get_water_depth, aggr='mean',
                                                           hec_ras_path=paths[self.EVENT_FILE_KEYS[0]],
                                                           nodes_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
            "water_volume": lambda: self._get_event_dynamic(event_idx, get_clipped_water_volume, aggr='mean',
                                                            hec_ras_path=paths[self.EVENT_FILE_KEYS[0]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self, event_idx: int) -> ndarray:
        paths = self._get_event_file_paths(event_idx)
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_event_dynamic(event_idx, get_velocity, aggr='mean',
                                                        filepath=paths[self.EVENT_FILE_KEYS[0]]),
            "face_flow": lambda: self._get_event_dynamic(event_idx, get_face_flow, aggr='mean',
                                                         filepath=paths[self.EVENT_FILE_KEYS[0]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_event_dynamic(self, event_idx: int, retrieval_func: Callable, aggr: str = 'first', **kwargs) -> ndarray:
        event_data = retrieval_func(**kwargs)
        event_data = self._get_trimmed_dynamic_data(event_data, event_idx, aggr)
        return event_data

    def _get_trimmed_dynamic_data(self, dynamic_data: ndarray, event_idx: int, aggr: str = 'first') -> ndarray:
        start = 0
        if self.spin_up_time is not None:
            if isinstance(self.spin_up_time, int):
                start = self.spin_up_time // self._event_base_timestep_interval[event_idx]
            elif isinstance(self.spin_up_time, dict):
                run_id = self.event_run_ids[event_idx]
                if run_id not in self.spin_up_time:
                    if 'default' in self.spin_up_time:
                        run_id = 'default'
                    else:
                        self.log_func(f'WARNING: No spin-up timesteps defined for Run ID {run_id} and no default value in dict. Setting start to 0.')
                start = self.spin_up_time.get(run_id, 0) // self._event_base_timestep_interval[event_idx]
            else:
                raise ValueError(f'Invalid type for spin_up_time: {type(self.spin_up_time)}')

        end = None
        if self.time_from_peak is not None:
            event_peak = self._event_peak_idx[event_idx]
            timesteps_from_peak = self.time_from_peak // self._event_base_timestep_interval[event_idx]
            end = event_peak + timesteps_from_peak

        trimmed = dynamic_data[start:end]

        step = self.timestep_interval // self._event_base_timestep_interval[event_idx]
        downsampled = self._downsample_dynamic_data(trimmed, step, aggr)

        return downsampled

    def _downsample_dynamic_data(self, dynamic_data: ndarray, step: int, aggr: str = 'first') -> ndarray:
        if step == 1:
            return dynamic_data

        # Trim array to be divisible by step
        trimmed_length = (dynamic_data.shape[0] // step) * step
        trimmed_array = dynamic_data[:trimmed_length]

        if aggr == 'first':
            return trimmed_array[::step]

        elif aggr in ['mean', 'sum']:
            # Reshape to group consecutive elements
            if dynamic_data.ndim == 1:
                reshaped = trimmed_array.reshape(-1, step) # (timesteps, step)
            else:
                reshaped = trimmed_array.reshape(-1, step, dynamic_data.shape[1]) # (timesteps, step, feature)

            if aggr == 'mean':
                return np.mean(reshaped, axis=1)
            elif aggr == 'sum':
                return np.sum(reshaped, axis=1)

        raise ValueError(f"Aggregation method '{aggr}' is not supported")

    def _get_features(self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]) -> List:
        features = []
        for feature in feature_list: # Order in feature list determines the order of features in the output
            if feature not in feature_retrieval_map:
                continue

            feature_data: ndarray = feature_retrieval_map[feature]()
            features.append(feature_data)

        return features

    def _get_physics_info(self, dynamic_nodes: ndarray) -> ndarray:
        # Denormalized Rainfall
        rainfall_idx = self.DYNAMIC_NODE_FEATURES.index('rainfall')
        node_rainfall_per_ts = dynamic_nodes[:, :, rainfall_idx]

        return node_rainfall_per_ts

    # =========== get() methods ===========

    def _get_node_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_NODE_FEATURES, timestep_idx)
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, timestep_idx)
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_timestep_dynamic_features(self, dynamic_features: ndarray, dynamic_feature_list: List[str], timestep_idx: int) -> Tensor:
        _, num_elems, _ = dynamic_features.shape
        if timestep_idx < self.previous_timesteps:
            # Pad with zeros if not enough previous timesteps are available; Currently not being used
            padding = self._get_empty_feature_tensor(dynamic_feature_list,
                                                     (self.previous_timesteps - timestep_idx, num_elems),
                                                     dtype=dynamic_features.dtype)
            ts_dynamic_features = np.concat([padding, dynamic_features[:timestep_idx+1, :, :]], axis=0)
        else:
            ts_dynamic_features = dynamic_features[timestep_idx-self.previous_timesteps:timestep_idx+1, :, :]
        return ts_dynamic_features

    def _get_timestep_features(self, static_features: ndarray, ts_dynamic_features: ndarray) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, dynamic_features (previous, current)]"""
        _, num_elems, _ = ts_dynamic_features.shape

        # (num_elems,  num_dynamic_features * num_timesteps)
        ts_dynamic_features = ts_dynamic_features.transpose(1, 0, 2)
        ts_dynamic_features = np.reshape(ts_dynamic_features, shape=(num_elems, -1), order='F')

        ts_data = np.concat([static_features, ts_dynamic_features], axis=1)
        return torch.from_numpy(ts_data)

    def _get_empty_feature_tensor(self, features: List[str], other_dims: Tuple[int, ...], dtype: np.dtype = np.float32) -> ndarray:
        if not self.is_normalized:
            return np.zeros((*other_dims, len(features)), dtype=dtype)
        return self.normalizer.get_normalized_zero_tensor(features, other_dims, dtype)

    def _get_timestep_labels(self, node_dynamic_features: ndarray, edge_dynamic_features: ndarray, timestep_idx: int) -> Tuple[Tensor, Tensor]:
        label_nodes_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        # (num_nodes, 1)
        current_nodes = node_dynamic_features[timestep_idx, :, label_nodes_idx][:, None]
        next_nodes = node_dynamic_features[timestep_idx+1, :, label_nodes_idx][:, None]
        label_nodes = next_nodes - current_nodes
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        # (num_edges, 1)
        current_edges = edge_dynamic_features[timestep_idx, :, label_edges_idx][:, None]
        next_edges = edge_dynamic_features[timestep_idx+1, :, label_edges_idx][:, None]
        label_edges = next_edges - current_edges
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges
    
    def _get_global_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, event_idx: int, timestep_idx: int) -> Dict[str, Tensor]:
        event_bc = self.boundary_conditions[event_idx]
        non_boundary_nodes_mask = ~event_bc.boundary_nodes_mask
        total_rainfall = node_rainfall_per_ts[timestep_idx, non_boundary_nodes_mask].sum(keepdims=True)

        total_rainfall = torch.from_numpy(total_rainfall)
        inflow_edges_mask = torch.from_numpy(event_bc.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(event_bc.outflow_edges_mask)

        return {
            'total_rainfall': total_rainfall,
            'inflow_edges_mask': inflow_edges_mask,
            'outflow_edges_mask': outflow_edges_mask,
        }

    def _get_local_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, event_idx: int, timestep_idx: int) -> Dict[str, Tensor]:
        rainfall = node_rainfall_per_ts[timestep_idx]

        rainfall = torch.from_numpy(rainfall)

        return {
            'rainfall': rainfall,
        }
