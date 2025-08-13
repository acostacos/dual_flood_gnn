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

from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_roughness,\
    get_cumulative_rainfall, get_water_level, get_water_volume, get_edge_direction_x,\
    get_edge_direction_y, get_face_length, get_velocity, get_face_flow
from .shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length, get_edge_slope
from .boundary_condition import BoundaryCondition
from .dataset_normalizer import DatasetNormalizer

class FloodEventDataset(Dataset):
    STATIC_NODE_FEATURES = ['area', 'roughness', 'elevation']
    DYNAMIC_NODE_FEATURES = ['rainfall', 'water_volume'] # Not included: 'water_depth'
    STATIC_EDGE_FEATURES = ['face_length', 'length', 'slope']
    DYNAMIC_EDGE_FEATURES = ['face_flow'] # Not included: 'velocity'
    NODE_TARGET_FEATURE = 'water_volume'
    EDGE_TARGET_FEATURE = 'face_flow'
    BASE_TIMESTEP_INTERVAL = 30  # in seconds
    BOUNDARY_CONDITION_NPZ_FILE = 'boundary_condition_masks.npz'
    CONSTANT_VALUES_NPZ_FILE = 'constant_values.npz'

    def __init__(self,
                 mode: Literal['train', 'test'],
                 root_dir: str,
                 dataset_summary_file: str,
                 nodes_shp_file: str,
                 edges_shp_file: str,
                 event_stats_file: str = 'event_stats.yaml',
                 features_stats_file: str = 'features_stats.yaml',
                 previous_timesteps: int = 2,
                 normalize: bool = True,
                 timestep_interval: int = BASE_TIMESTEP_INTERVAL,
                 spin_up_timesteps: Union[int, Dict[str, int]] = None,
                 timesteps_from_peak: Optional[int] = None,
                 inflow_boundary_nodes: List[int] = [],
                 outflow_boundary_nodes: List[int] = [],
                 with_global_mass_loss: bool = True,
                 with_local_mass_loss: bool = True,
                 debug: bool = False,
                 logger: Optional[Logger] = None,
                 force_reload: bool = False):
        assert mode in ['train', 'test'], f'Invalid mode: {mode}. Must be "train" or "test".'
        assert timestep_interval % self.BASE_TIMESTEP_INTERVAL == 0, f'Timestep interval must be a multiple of {self.BASE_TIMESTEP_INTERVAL} seconds.'

        self.log_func = print
        if logger is not None and hasattr(logger, 'log'):
            self.log_func = logger.log

        # File paths
        self.hec_ras_files, self.hec_ras_run_ids = self._get_hecras_files_from_summary(root_dir, dataset_summary_file)
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file
        self.event_stats_file = event_stats_file
        self.features_stats_file = features_stats_file

        # Dataset configurations
        self.mode = mode
        self.previous_timesteps = previous_timesteps
        self.is_normalized = normalize
        self.timestep_interval = timestep_interval
        self.spin_up_timesteps = spin_up_timesteps
        self.timesteps_from_peak = timesteps_from_peak
        self.inflow_boundary_nodes = inflow_boundary_nodes
        self.outflow_boundary_nodes = outflow_boundary_nodes
        self.with_global_mass_loss = with_global_mass_loss
        self.with_local_mass_loss = with_local_mass_loss

        # Dataset variables
        self.num_static_node_features = len(self.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(self.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(self.DYNAMIC_EDGE_FEATURES)
        self.event_start_idx, self.total_rollout_timesteps = self._load_event_stats(root_dir, event_stats_file)
        self._event_peak_idx = None
        self._event_num_timesteps = None

        # Helper classes
        self.normalizer = DatasetNormalizer(mode, root_dir, features_stats_file)
        self.boundary_condition = BoundaryCondition(root_dir=root_dir,
                                                    hec_ras_file=self.hec_ras_files[0],
                                                    inflow_boundary_nodes=self.inflow_boundary_nodes,
                                                    outflow_boundary_nodes=self.outflow_boundary_nodes,
                                                    saved_npz_file=self.BOUNDARY_CONDITION_NPZ_FILE)

        super().__init__(root_dir, transform=None, pre_transform=None, pre_filter=None, log=debug, force_reload=force_reload)


    @property
    def raw_file_names(self):
        return [self.nodes_shp_file, self.edges_shp_file, *self.hec_ras_files]

    @property
    def processed_file_names(self):
        dynamic_files = [f'dynamic_values_event_{run_id}.npz' for run_id in self.hec_ras_run_ids]
        return [
            self.event_stats_file,
            self.features_stats_file,
            self.BOUNDARY_CONDITION_NPZ_FILE,
            self.CONSTANT_VALUES_NPZ_FILE,
            *dynamic_files,
        ]

    def download(self):
        # Data must be downloaded manually and placed in the raw dir
        pass

    def process(self):
        self.log_func('Processing Flood Event Dataset...')

        all_event_timesteps = self._set_event_properties()
        edge_index = self._get_edge_index()

        static_nodes = self._get_static_node_features()
        dynamic_nodes = self._get_dynamic_node_features()
        static_edges = self._get_static_edge_features()
        dynamic_edges = self._get_dynamic_edge_features()

        self.boundary_condition.create(edge_index, dynamic_nodes, dynamic_edges)
        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = self.boundary_condition.remove(
            static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index,
        )

        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = self.boundary_condition.apply(
            static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index,
        )

        # Global Mass Loss Features
        global_mass_info = self._get_global_mass_info(dynamic_nodes, dynamic_edges)
        total_rainfall_per_ts, edge_face_flow_per_ts = global_mass_info

        # Local Mass Loss Features
        local_mass_info = self._get_local_mass_loss_info(dynamic_nodes, dynamic_edges)
        node_rainfall_per_ts, edge_face_flow_per_ts = local_mass_info

        if self.is_normalized:
            static_nodes = self.normalizer.normalize_feature_vector(self.STATIC_NODE_FEATURES, static_nodes)
            dynamic_nodes = self.normalizer.normalize_feature_vector(self.DYNAMIC_NODE_FEATURES, dynamic_nodes)
            static_edges = self.normalizer.normalize_feature_vector(self.STATIC_EDGE_FEATURES, static_edges)
            dynamic_edges = self.normalizer.normalize_feature_vector(self.DYNAMIC_EDGE_FEATURES, dynamic_edges)

        np.savez(self.processed_paths[3],
                 edge_index=edge_index,
                 static_nodes=static_nodes,
                 static_edges=static_edges)
        self.log_func(f'Saved constant values to {self.processed_paths[3]}')

        start_idx = 0
        for i, run_id in enumerate(self.hec_ras_run_ids):
            end_idx = start_idx + self._event_num_timesteps[i]

            event_timesteps = all_event_timesteps[start_idx:end_idx].copy()
            event_dynamic_nodes = dynamic_nodes[start_idx:end_idx].copy()
            event_dynamic_edges = dynamic_edges[start_idx:end_idx].copy()

            # Global Mass Conservation Features
            event_total_rainfall_per_ts = total_rainfall_per_ts[start_idx:end_idx].copy()

            # Local Mass Conservation Features
            event_rainfall_per_ts = node_rainfall_per_ts[start_idx:end_idx].copy()

            event_face_flow_per_ts = edge_face_flow_per_ts[start_idx:end_idx].copy()

            save_path = self.processed_paths[i + 4]
            np.savez(save_path,
                     event_timesteps=event_timesteps,
                     dynamic_nodes=event_dynamic_nodes,
                     dynamic_edges=event_dynamic_edges,
                     total_rainfall_per_ts=event_total_rainfall_per_ts,
                     node_rainfall_per_ts=event_rainfall_per_ts,
                     edge_face_flow_per_ts=event_face_flow_per_ts)
            self.log_func(f'Saved dynamic values for event {run_id} to {save_path}')

            start_idx = end_idx

        self._save_event_stats()
        self.log_func(f'Saved event stats to {self.processed_paths[0]}')
        if self.mode == 'train':
            self.normalizer.save_feature_stats()
            self.log_func(f'Saved feature stats to {self.processed_paths[1]}')
        self.boundary_condition.save_data()
        self.log_func(f'Saved boundary condition info to {self.processed_paths[2]}')

    def len(self):
        return self.total_rollout_timesteps

    def get(self, idx):
        # Load constant data
        constant_values = np.load(self.processed_paths[3])
        edge_index: ndarray = constant_values['edge_index']
        static_nodes: ndarray = constant_values['static_nodes']
        static_edges: ndarray = constant_values['static_edges']

        # Find the event this index belongs to using the start indices
        if idx < 0 or idx >= self.total_rollout_timesteps:
            raise IndexError(f'Index {idx} out of bounds for dataset with {self.total_rollout_timesteps} timesteps.')
        start_idx = 0
        for si in self.event_start_idx:
            if idx < si:
                break
            start_idx = si
        event_idx = self.event_start_idx.index(start_idx)

        # Load dynamic data
        dynamic_values_path = self.processed_paths[event_idx + 4]
        dynamic_values = np.load(dynamic_values_path)
        event_timesteps: ndarray = dynamic_values['event_timesteps']
        dynamic_nodes: ndarray = dynamic_values['dynamic_nodes']
        dynamic_edges: ndarray = dynamic_values['dynamic_edges']

        # Create Data object for timestep
        within_event_idx = idx - start_idx + self.previous_timesteps # First timestep starts at self.previous_timesteps
        timestep = event_timesteps[within_event_idx]
        node_features = self._get_node_timestep_data(static_nodes, dynamic_nodes, within_event_idx)
        edge_features = self._get_edge_timestep_data(static_edges, dynamic_edges, edge_index, within_event_idx)
        label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, within_event_idx)

        # Get physics-informed loss information
        global_mass_info = None
        if self.with_global_mass_loss:
            total_rainfall_per_ts: ndarray = dynamic_values['total_rainfall_per_ts']
            event_face_flow_per_ts: ndarray = dynamic_values['event_face_flow_per_ts']
            global_mass_info = self._get_global_mass_info_for_timestep(total_rainfall_per_ts,
                                                                       event_face_flow_per_ts,
                                                                       within_event_idx)

        local_mass_info = None
        if self.with_local_mass_loss:
            node_rainfall_per_ts: ndarray = dynamic_values['node_rainfall_per_ts']
            edge_face_flow_per_ts: ndarray = dynamic_values['edge_face_flow_per_ts']
            local_mass_info = self._get_local_mass_info_for_timestep(node_rainfall_per_ts,
                                                                     edge_face_flow_per_ts,
                                                                     within_event_idx)

        edge_index = torch.from_numpy(edge_index)
        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_features,
                    y=label_nodes,
                    y_edge=label_edges,
                    timestep=timestep,
                    global_mass_info=global_mass_info,
                    local_mass_info=local_mass_info)

        return data

    def _load_event_stats(self, root_dir: str, event_stats_file: str) -> Tuple[List[int], int]:
        event_stats_path = os.path.join(root_dir, 'processed', event_stats_file)
        if not os.path.exists(event_stats_path):
            return [], 0

        event_stats = read_yaml_file(event_stats_path)
        event_start_idx = event_stats['event_start_idx']
        total_rollout_timesteps = event_stats['total_rollout_timesteps']
        return event_start_idx, total_rollout_timesteps

    def _save_event_stats(self):
        event_stats = {
            'event_start_idx': self.event_start_idx,
            'total_rollout_timesteps': self.total_rollout_timesteps,
        }
        save_to_yaml_file(self.processed_paths[0], event_stats)

    def _get_hecras_files_from_summary(self, root_dir: str, dataset_summary_file: str) -> Tuple[List[str], List[str]]:
        '''Assumes all HEC-RAS files in the dataset summary are from the same catchment'''
        dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

        hec_ras_run_ids = []
        hec_ras_files = []
        for _, row in summary_df.iterrows():
            run_id = row['Run_ID']
            hec_ras_path = row['HECRAS_Filepath']

            assert run_id not in hec_ras_run_ids, f'Duplicate Run_ID found: {run_id}'
            full_hec_ras_path = os.path.join(root_dir, 'raw', hec_ras_path)
            assert os.path.exists(full_hec_ras_path), f'HECRAS file not found: {hec_ras_path}'

            hec_ras_run_ids.append(run_id)
            hec_ras_files.append(hec_ras_path)

        return hec_ras_files, hec_ras_run_ids

    # =========== process() methods ===========

    def _set_event_properties(self) -> ndarray:
        self._event_peak_idx = []
        self._event_num_timesteps = []
        self.event_start_idx = []

        current_total_ts = 0
        all_event_timesteps = []
        for event_idx, hec_ras_path in enumerate(self.raw_paths[2:]):
            water_volume = get_water_volume(hec_ras_path)
            total_water_volume = water_volume.sum(axis=1)
            peak_idx = np.argmax(total_water_volume)
            self._event_peak_idx.append(peak_idx)

            timesteps = get_event_timesteps(hec_ras_path)
            timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx)
            all_event_timesteps.append(timesteps)

            num_timesteps = len(timesteps)
            self._event_num_timesteps.append(num_timesteps)

            event_total_rollout_ts = num_timesteps - self.previous_timesteps - 1  # First timestep starts at self.previous_timesteps; Last timestep is used for labels
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        assert len(self._event_peak_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self._event_num_timesteps) == len(self.hec_ras_run_ids), 'Mismatch in number of events and number of timesteps.'
        assert len(self.event_start_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and start indices.'
        assert np.all((np.array(self._event_peak_idx) - (self.timesteps_from_peak if self.timesteps_from_peak is not None else 0)) >= 0),\
            'Timesteps from peak exceed available timesteps.'

        all_event_timesteps = np.concatenate(all_event_timesteps, axis=0)
        return all_event_timesteps

    def _get_edge_index(self) -> ndarray:
        edge_index = get_edge_index(self.raw_paths[1])
        return edge_index

    def _get_static_node_features(self) -> ndarray:
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(self.raw_paths[2]),
            "roughness": lambda: get_roughness(self.raw_paths[2]),
            "elevation": lambda: get_cell_elevation(self.raw_paths[0]),
        }

        static_features = self._get_features(feature_list=self.STATIC_NODE_FEATURES,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_static_edge_features(self) -> ndarray:
        STATIC_EDGE_RETRIEVAL_MAP = {
            "direction_x": lambda: get_edge_direction_x(self.raw_paths[2]),
            "direction_y": lambda: get_edge_direction_y(self.raw_paths[2]),
            "face_length": lambda: get_face_length(self.raw_paths[2]),
            "length": lambda: get_edge_length(self.raw_paths[1]),
            "slope": lambda: get_edge_slope(self.raw_paths[1]),
        }

        static_features = self._get_features(feature_list=self.STATIC_EDGE_FEATURES,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_dynamic_node_features(self) -> ndarray:
        def get_water_depth():
            """Get water depth from water level and elevation"""
            water_level = get_water_level(self.raw_paths[0])
            elevation = get_cell_elevation(self.raw_paths[0])[None, :]
            water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
            return water_depth

        def get_interval_rainfall(hec_ras_paths: List[str]):
            """Get interval rainfall from cumulative rainfall"""
            all_event_data = []
            for i, path in enumerate(hec_ras_paths):
                cumulative_rainfall = get_cumulative_rainfall(path)
                cumulative_rainfall = self._get_trimmed_dynamic_data(cumulative_rainfall, i)

                last_ts_rainfall = np.zeros((1, cumulative_rainfall.shape[1]), dtype=cumulative_rainfall.dtype)
                intervals = np.diff(cumulative_rainfall, axis=0)
                interval_rainfall = np.concatenate((intervals, last_ts_rainfall), axis=0)

                all_event_data.append(interval_rainfall)
            all_event_data = np.concatenate(all_event_data, axis=0)
            return all_event_data

        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "rainfall": lambda: get_interval_rainfall(self.raw_paths[2:]),
            "water_depth": lambda: self._get_dynamic_from_all_events(get_water_depth),
            "water_volume": lambda: self._get_dynamic_from_all_events(get_water_volume),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self) -> ndarray:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_events(get_velocity),
            "face_flow": lambda: self._get_dynamic_from_all_events(get_face_flow),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_from_all_events(self, retrieval_func: Callable) -> ndarray:
        all_event_data = []
        for i, hec_ras_path in enumerate(self.raw_paths[2:]):
            event_data = retrieval_func(hec_ras_path)
            event_data = self._get_trimmed_dynamic_data(event_data, i)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_trimmed_dynamic_data(self, dynamic_data: ndarray, event_idx: int) -> ndarray:
        start = 0
        if self.spin_up_timesteps is not None:
            if isinstance(self.spin_up_timesteps, int):
                start = self.spin_up_timesteps
            elif isinstance(self.spin_up_timesteps, dict):
                run_id = self.hec_ras_run_ids[event_idx]
                if run_id not in self.spin_up_timesteps:
                    self.log_func(f'No spin-up timesteps defined for Run ID {run_id}. Using default value of 0.')
                start = self.spin_up_timesteps.get(run_id, 0)
            else:
                raise ValueError(f'Invalid type for spin_up_timesteps: {type(self.spin_up_timesteps)}')

        end = None
        if self.timesteps_from_peak is not None:
            event_peak = self._event_peak_idx[event_idx]
            end = event_peak + self.timesteps_from_peak

        step = self.timestep_interval // self.BASE_TIMESTEP_INTERVAL

        return dynamic_data[start:end:step]

    def _get_features(self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]) -> List:
        features = []
        for feature in feature_list: # Order in feature list determines the order of features in the output
            if feature not in feature_retrieval_map:
                continue

            feature_data: ndarray = feature_retrieval_map[feature]()

            if self.mode == 'train' and self.is_normalized:
                self.normalizer.update_stats(feature, feature_data)

            features.append(feature_data)

        return features

    def _get_global_mass_info(self, dynamic_nodes: ndarray, dynamic_edges: ndarray) -> Tuple[ndarray, ndarray]:
        # Total rainfall 
        rainfall_idx = self.DYNAMIC_NODE_FEATURES.index('rainfall')
        non_boundary_nodes_mask = ~self.boundary_condition.boundary_nodes_mask
        node_rainfall_per_ts = dynamic_nodes[:, non_boundary_nodes_mask, rainfall_idx]
        total_rainfall_per_ts = node_rainfall_per_ts.sum(axis=1)

        # Normalized Face flow (w/ unmasked outflow values)
        face_flow_idx = self.DYNAMIC_EDGE_FEATURES.index('face_flow')
        edge_face_flow_per_ts = dynamic_edges[:, :, face_flow_idx]
        if self.is_normalized:
            mean, std = self.normalizer.get_feature_mean_std('face_flow')
            edge_face_flow_per_ts = self.normalizer.normalize(edge_face_flow_per_ts, mean, std)

        return total_rainfall_per_ts, edge_face_flow_per_ts

    def _get_local_mass_loss_info(self, dynamic_nodes: ndarray, dynamic_edges: ndarray) -> Tuple[ndarray, ndarray]:
        # Rainfall
        rainfall_idx = self.DYNAMIC_NODE_FEATURES.index('rainfall')
        non_boundary_nodes_mask = ~self.boundary_condition.boundary_nodes_mask
        node_rainfall_per_ts = dynamic_nodes[:, non_boundary_nodes_mask, rainfall_idx]

        # Normalized Face flow (w/ unmasked outflow values)
        face_flow_idx = self.DYNAMIC_EDGE_FEATURES.index('face_flow')
        edge_face_flow_per_ts = dynamic_edges[:, :, face_flow_idx]
        if self.is_normalized:
            mean, std = self.normalizer.get_feature_mean_std('face_flow')
            edge_face_flow_per_ts = self.normalizer.normalize(edge_face_flow_per_ts, mean, std)

        return node_rainfall_per_ts, edge_face_flow_per_ts

    # =========== get() methods ===========

    def _get_node_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_NODE_FEATURES, timestep_idx)

        # Mask node boundary conditions = only keep outflow water volume
        num_ts, _, _ = ts_dynamic_features.shape
        outflow_boundary_nodes = self.boundary_condition.new_outflow_boundary_nodes
        boundary_nodes = self.boundary_condition.get_new_boundary_nodes()
        target_nodes_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)

        masked_boundary_dynamic_nodes = self._get_empty_feature_tensor(features=self.DYNAMIC_NODE_FEATURES,
                                                                       other_dims=(num_ts, len(boundary_nodes)),
                                                                       dtype=ts_dynamic_features.dtype)

        outflow_dynamic_nodes = ts_dynamic_features[:, outflow_boundary_nodes, :].copy()
        nodes_overwrite_mask = np.isin(boundary_nodes, outflow_boundary_nodes)
        masked_boundary_dynamic_nodes[:, nodes_overwrite_mask, target_nodes_idx] = outflow_dynamic_nodes[:, :, target_nodes_idx]

        boundary_nodes_mask = self.boundary_condition.boundary_nodes_mask
        ts_dynamic_features = np.concat([ts_dynamic_features[:, ~boundary_nodes_mask, :], masked_boundary_dynamic_nodes], axis=1)
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, edge_index: ndarray, timestep_idx: int) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, timestep_idx)

        # Mask edge boundary conditions = only keep inflow water flow
        num_ts, _, _ = ts_dynamic_features.shape
        inflow_edges_mask = self.boundary_condition.inflow_edges_mask
        inflow_boundary_nodes = self.boundary_condition.new_inflow_boundary_nodes
        target_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)

        boundary_edges_mask = self.boundary_condition.boundary_edges_mask
        num_boundary_edges = boundary_edges_mask.sum()
        masked_boundary_dynamic_edges = self._get_empty_feature_tensor(features=self.DYNAMIC_EDGE_FEATURES,
                                                                       other_dims=(num_ts, num_boundary_edges),
                                                                       dtype=ts_dynamic_features.dtype)

        inflow_dynamic_edges = ts_dynamic_features[:, inflow_edges_mask, :].copy()
        edges_overwrite_mask = np.any(np.isin(edge_index[:, boundary_edges_mask], inflow_boundary_nodes), axis=0)
        masked_boundary_dynamic_edges[:, edges_overwrite_mask, target_edges_idx] = inflow_dynamic_edges[:, :, target_edges_idx]

        ts_dynamic_features = np.concat([ts_dynamic_features[:, ~boundary_edges_mask, :], masked_boundary_dynamic_edges], axis=1)

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
        label_nodes = node_dynamic_features[timestep_idx+1, :, label_nodes_idx][:, None]
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        # (num_nodes, 1)
        label_edges = edge_dynamic_features[timestep_idx+1, :, label_edges_idx][:, None]
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges
    
    def _get_global_mass_info_for_timestep(self,
                                           total_rainfall_per_ts: ndarray,
                                           edge_face_flow_per_ts: ndarray,
                                           timestep_idx: int) -> Dict[str, Tensor]:
        total_rainfall = torch.from_numpy(total_rainfall_per_ts[[timestep_idx]])
        face_flow = torch.from_numpy(edge_face_flow_per_ts[timestep_idx][:, None])
        inflow_edges_mask = torch.from_numpy(self.boundary_condition.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(self.boundary_condition.outflow_edges_mask)
        non_boundary_nodes_mask = torch.from_numpy(~self.boundary_condition.boundary_nodes_mask)

        return {
            'total_rainfall': total_rainfall,
            'face_flow': face_flow,
            'inflow_edges_mask': inflow_edges_mask,
            'outflow_edges_mask': outflow_edges_mask,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }

    def _get_local_mass_info_for_timestep(self,
                                          node_rainfall_per_ts: ndarray,
                                          edge_face_flow_per_ts: ndarray,
                                          timestep_idx: int) -> Dict[str, Tensor]:
        rainfall = torch.from_numpy(node_rainfall_per_ts[timestep_idx])
        face_flow = torch.from_numpy(edge_face_flow_per_ts[timestep_idx][:, None])
        non_boundary_nodes_mask = torch.from_numpy(~self.boundary_condition.boundary_nodes_mask)

        return {
            'rainfall': rainfall,
            'face_flow': face_flow,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }
