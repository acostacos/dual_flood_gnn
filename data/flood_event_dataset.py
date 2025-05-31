import os
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, Tuple, List, Dict, Optional
from utils.logger import Logger
from utils.file_utils import read_yaml_file, save_to_yaml_file

from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_roughness, get_rainfall,\
    get_water_level, get_water_volume, get_edge_direction_x, get_edge_direction_y, get_face_length,\
    get_velocity, get_face_flow
from .shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length, get_edge_slope

class FloodEventDataset(InMemoryDataset):
    STATIC_NODE_FEATURES = ['area', 'roughness', 'elevation']
    DYNAMIC_NODE_FEATURES = ['rainfall', 'water_volume'] # Not included: 'water_depth'
    STATIC_EDGE_FEATURES = ['face_length', 'length', 'slope']
    DYNAMIC_EDGE_FEATURES = ['face_flow'] # Not included: 'velocity'
    NODE_TARGET_FEATURE = 'water_volume'
    EDGE_TARGET_FEATURE = 'face_flow'

    def __init__(self,
                 root_dir: str,
                 dataset_summary_file: str,
                 nodes_shp_file: str,
                 edges_shp_file: str,
                 feature_stats_file: str,

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
        self.hec_ras_files = self._get_hecras_files_from_summary(root_dir, dataset_summary_file)
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file
        self.feature_stats_file = feature_stats_file

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
        self.feature_stats = {}

        super().__init__(root_dir, transform, pre_transform, pre_filter, log=debug)

        self.load(self.processed_paths[0])
        if len(self.feature_stats) == 0:
            self.feature_stats = self.get_feature_stats()

    @property
    def raw_file_names(self):
        return [self.nodes_shp_file, self.edges_shp_file, *self.hec_ras_files]

    @property
    def processed_file_names(self):
        return ['complete_data.pt', self.feature_stats_file]

    def download(self):
        # Data must be downloaded manually and placed in the raw_dir
        pass

    def process(self):
        self.log_func('Processing Flood Event Dataset...')

        edge_index, self.event_start_idx = self._get_graph_properties()
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

        dataset = []
        total_num_timesteps = next(iter(dynamic_nodes.values())).shape[0]
        for i in range(total_num_timesteps - 1):
            node_feature_list = FloodEventDataset.STATIC_NODE_FEATURES + FloodEventDataset.DYNAMIC_NODE_FEATURES
            node_features = self._get_timestep_data(node_feature_list, static_nodes, dynamic_nodes, i)

            edge_feature_list = FloodEventDataset.STATIC_EDGE_FEATURES + FloodEventDataset.DYNAMIC_EDGE_FEATURES
            edge_features = self._get_timestep_data(edge_feature_list, static_edges, dynamic_edges, i)

            label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, i)

            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label_nodes, y_edge=label_edges)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            dataset.append(data)

        self.save(dataset, self.processed_paths[0])

        self.save_feature_stats()

    def get_feature_stats(self) -> Dict:
        if not os.path.exists(self.processed_paths[1]):
            return {}

        feature_stats = read_yaml_file(self.processed_paths[1])
        return feature_stats

    def save_feature_stats(self):
        save_to_yaml_file(self.processed_paths[1], self.feature_stats)

    # =========== Helper Methods ===========

    def _get_hecras_files_from_summary(self, root_dir: str, dataset_summary_file: str) -> List[str]:
        '''Assumes all HEC-RAS files in the dataset summary are from the same catchment'''
        dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

        hec_ras_files = []
        for _, row in summary_df.iterrows():
            hec_ras_path = row['HECRAS_Filepath']

            full_hec_ras_path = os.path.join(root_dir, 'raw', hec_ras_path)
            assert os.path.exists(full_hec_ras_path), f'HECRAS file not found: {hec_ras_path}'

            hec_ras_files.append(hec_ras_path)

        return hec_ras_files

    def _get_graph_properties(self) -> Tuple[torch.Tensor, List[int]]:
        edge_index = get_edge_index(self.raw_paths[1])
        edge_index = torch.from_numpy(edge_index)

        event_start_idx = []
        current_total_ts = 0
        for hec_ras_path in self.raw_paths[2:]:
            num_timesteps = len(get_event_timesteps(hec_ras_path))
            event_start_idx.append(current_total_ts)
            current_total_ts += num_timesteps

        return edge_index

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

    def _get_timestep_data(self, feature_list: str, static_features: Dict[str, np.ndarray], dynamic_features: Dict[str, np.ndarray], timestep_idx: int) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, previous_dynamic_features, current_dynamic_features]"""

        ordered_static_feature_list = [k for k in feature_list if k in static_features.keys()]
        ordered_dynamic_feature_list = [k for k in feature_list if k in dynamic_features.keys()]

        ts_static_features = self._get_static_timestep_data(static_features, ordered_static_feature_list)
        ts_dynamic_features = self._get_dynamic_timestep_data(dynamic_features, ordered_dynamic_feature_list, timestep_idx)

        return torch.cat([ts_static_features, ts_dynamic_features], dim=1)

    def _get_static_timestep_data(self, static_features: Dict[str, np.ndarray], feature_order: List[str]) -> Tensor:
        """Returns the static features for the timestep in the shape [num_items, num_features]"""
        ts_static_features = [static_features[feature] for feature in feature_order]
        ts_static_features = np.array(ts_static_features).transpose()
        return torch.from_numpy(ts_static_features)

    def _get_dynamic_timestep_data(self, dynamic_features: Dict[str, np.ndarray], feature_order: List[str], timestep_idx: int) -> Tensor:
        """Returns the dynamic features for the timestep in the shape [num_items, num_features]. Includes the current timestep and previous timesteps."""
        ts_dynamic_features = []
        for feature in feature_order:
            for i in range(self.previous_timesteps, -1, -1):
                if timestep_idx-i < 0:
                    # Pad with zeros if no previous data is available
                    ts_dynamic_features.append(np.zeros_like(dynamic_features[feature][0]))
                else:
                    ts_dynamic_features.append(dynamic_features[feature][timestep_idx-i])
        ts_dynamic_features = np.array(ts_dynamic_features).transpose()

        return torch.from_numpy(ts_dynamic_features)
    
    def _get_timestep_labels(self, node_dynamic_features: Dict[str, np.ndarray], edge_dynamic_features: Dict[str, np.ndarray], timestep_idx: int) -> Tuple[Tensor, Tensor]:
        label_nodes = node_dynamic_features[FloodEventDataset.NODE_TARGET_FEATURE][timestep_idx+1]
        label_nodes = label_nodes[:, None] # Reshape to [num_nodes, 1]
        label_nodes = torch.from_numpy(label_nodes)

        label_edges = edge_dynamic_features[FloodEventDataset.EDGE_TARGET_FEATURE][timestep_idx+1]
        label_edges = label_edges[:, None] # Reshape to [num_edges, 1]
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges

    # =========== Feature Retrieval Methods ===========

    def _get_static_node_features(self) -> Dict[str, np.ndarray]:
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(self.raw_paths[2]),
            "roughness": lambda: get_roughness(self.raw_paths[2]),
            "elevation": lambda: get_cell_elevation(self.raw_paths[0]),
        }

        return self._get_features(feature_list=FloodEventDataset.STATIC_NODE_FEATURES,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP)

    def _get_static_edge_features(self) -> Dict[str, np.ndarray]:
        STATIC_EDGE_RETRIEVAL_MAP = {
            "direction_x": lambda: get_edge_direction_x(self.raw_paths[2]),
            "direction_y": lambda: get_edge_direction_y(self.raw_paths[2]),
            "face_length": lambda: get_face_length(self.raw_paths[2]),
            "length": lambda: get_edge_length(self.raw_paths[1]),
            "slope": lambda: get_edge_slope(self.raw_paths[1]),
        }

        return self._get_features(feature_list=FloodEventDataset.STATIC_EDGE_FEATURES,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP)

    def _get_dynamic_node_features(self) -> Dict[str, np.ndarray]:
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

        return self._get_features(feature_list=FloodEventDataset.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)

    def _get_dynamic_edge_features(self) -> Dict[str, np.ndarray]:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_events(get_velocity),
            "face_flow": lambda: self._get_dynamic_from_all_events(get_face_flow),
        }

        return self._get_features(feature_list=FloodEventDataset.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)

    def _get_dynamic_from_all_events(self, retrieval_func: Callable) -> np.ndarray:
        all_event_data = []
        for hec_ras_path in self.raw_paths[2:]:
            event_data = retrieval_func(hec_ras_path)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_features(self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]) -> Dict[str, np.ndarray]:
        features = {}
        for feature in feature_list:
            if feature not in feature_retrieval_map:
                continue

            feature_data: np.ndarray = feature_retrieval_map[feature]()
            self.feature_stats[feature] = {
                'mean': feature_data.mean().item(),
                'std': feature_data.std().item(),
            }

            if self.normalize:
                feature_data = self._normalize_features(feature_data)

            features[feature] = feature_data

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
