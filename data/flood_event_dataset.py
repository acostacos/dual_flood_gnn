import os
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Callable, Tuple, List, Dict, Optional
from utils.logger import Logger
from utils.file_utils import read_yaml_file, save_to_yaml_file

from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_roughness, get_rainfall,\
    get_water_level, get_water_volume, get_edge_direction_x, get_edge_direction_y, get_face_length,\
    get_velocity, get_face_flow
from .shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length, get_edge_slope

class FloodEventDataset(Dataset):
    STATIC_NODE_FEATURES = ['area', 'roughness', 'elevation']
    DYNAMIC_NODE_FEATURES = ['rainfall', 'water_volume'] # Not included: 'water_depth'
    STATIC_EDGE_FEATURES = ['face_length', 'length', 'slope']
    DYNAMIC_EDGE_FEATURES = ['face_flow'] # Not included: 'velocity'
    NODE_TARGET_FEATURE = DYNAMIC_NODE_FEATURES[-1] # 'water_volume'
    EDGE_TARGET_FEATURE = DYNAMIC_EDGE_FEATURES[-1] # 'face_flow'

    def __init__(self,
                 root_dir: str,
                 dataset_summary_file: str,
                 nodes_shp_file: str,
                 edges_shp_file: str,
                 previous_timesteps: int = 2,
                 normalize: bool = True,
                 timesteps_from_peak: Optional[int] = None,
                 debug: bool = False,
                 logger: Logger = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.log_func = print
        if logger is not None and hasattr(logger, 'log'):
            self.log_func = logger.log

        # File paths
        self.hec_ras_files, self.hec_ras_run_ids = self._get_hecras_files_from_summary(root_dir, dataset_summary_file)
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file

        # Dataset configurations
        self.normalize = normalize
        self.previous_timesteps = previous_timesteps
        self.timesteps_from_peak = timesteps_from_peak

        # Dataset variables
        self.num_static_node_features = len(FloodEventDataset.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(FloodEventDataset.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(FloodEventDataset.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(FloodEventDataset.DYNAMIC_EDGE_FEATURES)
        self.event_start_idx = []
        self.total_train_timesteps = 0
        self.feature_stats = {}

        super().__init__(root_dir, transform, pre_transform, pre_filter, log=debug)

        if len(self.feature_stats) == 0:
            self.feature_stats = self.load_feature_stats()
        if len(self.event_start_idx) == 0 or self.total_train_timesteps == 0:
            self.event_start_idx, self.total_train_timesteps = self.load_event_stats()

    @property
    def raw_file_names(self):
        return [self.nodes_shp_file, self.edges_shp_file, *self.hec_ras_files]

    @property
    def processed_file_names(self):
        dynamic_node_files = [f'dynamic_node_event_{run_id}.pt' for run_id in self.hec_ras_run_ids]
        dynamic_edge_files = [f'dynamic_edge_event_{run_id}.pt' for run_id in self.hec_ras_run_ids]
        return ['event_stats.yaml', 'features_stats.yaml', 'edge_index.pt', 'static_node.pt', 'static_edge.pt',
                *dynamic_node_files, *dynamic_edge_files]

    def download(self):
        # Data must be downloaded manually and placed in the raw_dir
        pass

    def process(self):
        self.log_func('Processing Flood Event Dataset...')

        global_properties = self._get_global_properties()
        edge_index, self.event_start_idx, self.total_train_timesteps, event_num_timesteps = global_properties
        static_nodes = self._get_static_node_features()
        dynamic_nodes = self._get_dynamic_node_features()
        static_edges = self._get_static_edge_features()
        dynamic_edges = self._get_dynamic_edge_features()

        if self.timesteps_from_peak is not None:
            # Trim the data from the peak water level
            peak_water_depth_ts = dynamic_nodes['water_depth'].sum(axis=1).argmax()
            last_ts = peak_water_depth_ts + self.timesteps_from_peak
            dynamic_nodes = self._trim_features_from_peak_water_depth(dynamic_nodes, last_ts)
            dynamic_edges = self._trim_features_from_peak_water_depth(dynamic_edges, last_ts)

        torch.save(edge_index, self.processed_paths[2])
        self.log_func(f'Saved edge index to {self.processed_paths[2]}')
        torch.save(static_nodes, self.processed_paths[3])
        self.log_func(f'Saved static node features to {self.processed_paths[3]}')
        torch.save(static_edges, self.processed_paths[4])
        self.log_func(f'Saved static edge features to {self.processed_paths[4]}')

        start_idx = 0
        for i, num_ts in enumerate(event_num_timesteps):
            run_id = self.hec_ras_run_ids[i]
            end_idx = start_idx + num_ts

            event_dynamic_nodes = dynamic_nodes[start_idx:end_idx].clone()
            torch.save(event_dynamic_nodes, self.processed_paths[i+5])
            self.log_func(f'Saved dynamic node features for event {run_id} to {self.processed_paths[i + 5]}')

            event_dynamic_edges = dynamic_edges[start_idx:end_idx].clone()
            torch.save(event_dynamic_edges, self.processed_paths[i + len(self.hec_ras_run_ids) + 5])
            self.log_func(f'Saved dynamic node features for event {run_id} to {self.processed_paths[i + 5]}')

            start_idx = end_idx

        self.save_event_stats()
        self.log_func(f'Saved event stats to {self.processed_paths[0]}')
        self.save_feature_stats()
        self.log_func(f'Saved feature stats to {self.processed_paths[1]}')

    def len(self):
        return self.total_train_timesteps

    def get(self, idx):
        edge_index = torch.load(self.processed_paths[2], weights_only=True)
        static_nodes = torch.load(self.processed_paths[3], weights_only=True)
        static_edges = torch.load(self.processed_paths[4], weights_only=True)

        # Find the event this index belongs to using the start indices
        if idx < 0 or idx >= self.total_train_timesteps:
            raise IndexError(f'Index {idx} out of bounds for dataset with {self.total_train_timesteps} timesteps.')
        start_idx = 0
        for si in self.event_start_idx:
            if idx < si:
                break
            start_idx = si
        event_idx = self.event_start_idx.index(start_idx)

        # Create Data object for timestep
        dynamic_nodes = torch.load(self.processed_paths[event_idx + 5], weights_only=True)
        dynamic_edges = torch.load(self.processed_paths[event_idx + len(self.hec_ras_run_ids) + 5], weights_only=True)

        within_event_idx = idx - start_idx

        node_features = self._get_timestep_data(static_nodes, dynamic_nodes, within_event_idx)
        edge_features = self._get_timestep_data(static_edges, dynamic_edges, within_event_idx)

        label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, within_event_idx)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label_nodes, y_edge=label_edges)

        return data

    def load_event_stats(self) -> Tuple[List[int], int]:
        if not os.path.exists(self.processed_paths[0]):
            return {}

        event_stats = read_yaml_file(self.processed_paths[0])
        event_start_idx = event_stats['event_start_idx']
        total_train_timesteps = event_stats['total_train_timesteps']
        return event_start_idx, total_train_timesteps

    def save_event_stats(self):
        event_stats = {
            'event_start_idx': self.event_start_idx,
            'total_train_timesteps': self.total_train_timesteps,
        }
        save_to_yaml_file(self.processed_paths[0], event_stats)

    def load_feature_stats(self) -> Dict:
        if not os.path.exists(self.processed_paths[1]):
            return {}

        feature_stats = read_yaml_file(self.processed_paths[1])
        return feature_stats

    def save_feature_stats(self):
        save_to_yaml_file(self.processed_paths[1], self.feature_stats)

    # =========== Helper Methods ===========

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

    def _get_global_properties(self) -> Tuple[Tensor, List[int], int, List[int]]:
        edge_index = get_edge_index(self.raw_paths[1])
        edge_index = torch.from_numpy(edge_index)

        event_start_idx = []
        event_num_timesteps = []
        current_total_ts = 0
        for hec_ras_path in self.raw_paths[2:]:
            num_timesteps = len(get_event_timesteps(hec_ras_path))
            event_num_timesteps.append(num_timesteps)

            event_total_training_ts = num_timesteps - 1  # Last timestep is used for labels
            event_start_idx.append(current_total_ts)
            current_total_ts += event_total_training_ts

        assert len(event_start_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and start indices.'

        return edge_index, event_start_idx, current_total_ts, event_num_timesteps

    def _trim_features_from_peak_water_depth(self, feature_data: Dict[str, np.ndarray], last_ts: int) -> Dict[str, np.ndarray]:
        for feature, data in feature_data.items():
            if self.normalize:
                data = self._denormalize_features(feature, data)

            data = data[:last_ts]
            self.feature_stats[feature] = {
                'mean': data.mean().item(),
                'std': data.std().item(),
            }

            if self.normalize:
                data = self._normalize_features(data)

            feature_data[feature] = data

        return feature_data

    def _get_timestep_data(self, static_features: Tensor, dynamic_features: Tensor, timestep_idx: int) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, previous_dynamic_features, current_dynamic_features]"""
        _, num_elems, num_dyn_features = dynamic_features.shape
        if timestep_idx < self.previous_timesteps:
            # Pad with zeros if not enough previous timesteps are available
            padding = torch.zeros((self.previous_timesteps - timestep_idx, num_elems, num_dyn_features), dtype=dynamic_features.dtype)
            ts_dynamic_features = torch.cat([padding, dynamic_features[:timestep_idx+1, :, :]], dim=0)
        else:
            ts_dynamic_features = dynamic_features[timestep_idx-self.previous_timesteps:timestep_idx+1, :, :]

        # (num_elems, num_timesteps * num_dynamic_features)
        ts_dynamic_features = ts_dynamic_features.transpose(0, 1).reshape(num_elems, -1)

        return torch.cat([static_features, ts_dynamic_features], dim=1)

    def _get_timestep_labels(self, node_dynamic_features: Tensor, edge_dynamic_features: Tensor, timestep_idx: int) -> Tuple[Tensor, Tensor]:
        # Target feature must be the last feature in the dynamic features
        label_nodes_idx = FloodEventDataset.DYNAMIC_NODE_FEATURES.index(FloodEventDataset.NODE_TARGET_FEATURE)
        # (num_nodes, 1)
        label_nodes = node_dynamic_features[timestep_idx+1, :, label_nodes_idx][:, None]

        label_edges_idx = FloodEventDataset.DYNAMIC_EDGE_FEATURES.index(FloodEventDataset.EDGE_TARGET_FEATURE)
        # (num_nodes, 1)
        label_edges = edge_dynamic_features[timestep_idx+1, :, label_edges_idx][:, None]

        return label_nodes, label_edges

    # =========== Feature Retrieval Methods ===========

    def _get_static_node_features(self) -> Tensor:
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(self.raw_paths[2]),
            "roughness": lambda: get_roughness(self.raw_paths[2]),
            "elevation": lambda: get_cell_elevation(self.raw_paths[0]),
        }

        static_features = self._get_features(feature_list=FloodEventDataset.STATIC_NODE_FEATURES,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return torch.from_numpy(static_features)

    def _get_static_edge_features(self) -> Tensor:
        STATIC_EDGE_RETRIEVAL_MAP = {
            "direction_x": lambda: get_edge_direction_x(self.raw_paths[2]),
            "direction_y": lambda: get_edge_direction_y(self.raw_paths[2]),
            "face_length": lambda: get_face_length(self.raw_paths[2]),
            "length": lambda: get_edge_length(self.raw_paths[1]),
            "slope": lambda: get_edge_slope(self.raw_paths[1]),
        }

        static_features = self._get_features(feature_list=FloodEventDataset.STATIC_EDGE_FEATURES,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        return torch.from_numpy(static_features)

    def _get_dynamic_node_features(self) -> Tensor:
        def get_water_depth():
            """Get water depth from water level and elevation"""
            water_level = get_water_level(self.raw_paths[0])
            elevation = get_cell_elevation(self.raw_paths[0])[None, :]
            water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
            return water_depth

        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "rainfall": lambda: self._get_dynamic_from_all_events(get_rainfall),
            "water_depth": lambda: self._get_dynamic_from_all_events(get_water_depth),
            "water_volume": lambda: self._get_dynamic_from_all_events(get_water_volume),
        }

        dynamic_features = self._get_features(feature_list=FloodEventDataset.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return torch.from_numpy(dynamic_features)

    def _get_dynamic_edge_features(self) -> Tensor:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_events(get_velocity),
            "face_flow": lambda: self._get_dynamic_from_all_events(get_face_flow),
        }

        dynamic_features = self._get_features(feature_list=FloodEventDataset.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return torch.from_numpy(dynamic_features)

    def _get_dynamic_from_all_events(self, retrieval_func: Callable) -> np.ndarray:
        all_event_data = []
        for hec_ras_path in self.raw_paths[2:]:
            event_data = retrieval_func(hec_ras_path)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_features(self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]) -> List:
        features = []
        for feature in feature_list: # Order in feature list determines the order of features in tensor
            if feature not in feature_retrieval_map:
                continue

            feature_data: np.ndarray = feature_retrieval_map[feature]()
            self.feature_stats[feature] = {
                'mean': feature_data.mean().item(),
                'std': feature_data.std().item(),
            }

            if self.normalize:
                feature_data = self._normalize_features(feature_data)

            features.append(feature_data)

        return features

    def _normalize_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Z-score normalization of features"""
        EPS = 1e-7 # Prevent division by zero
        return (feature_data - feature_data.mean()) / (feature_data.std() + EPS)

    def _denormalize_features(self, feature: str, feature_data: np.ndarray) -> np.ndarray:
        """Z-score denormalization of features"""
        if feature not in self.feature_stats:
            raise ValueError(f'Feature {feature} not found in feature stats.')

        EPS = 1e-7
        return feature_data * (self.feature_stats[feature]['std'] + EPS) + self.feature_stats[feature]['mean']
