import os
import torch
import numpy as np

from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, Tuple, List, Dict, Literal
from utils.logger import Logger

from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_roughness, get_rainfall, get_water_level,\
                                    get_edge_direction_x, get_edge_direction_y, get_face_length, get_velocity
from .shp_data_retrieval import get_edge_index, get_cell_elevation, get_edge_length, get_edge_slope

class InMemoryFloodEventDataset(InMemoryDataset):
    def __init__(self,
                 dataset_info_path: str,
                 root_dir: str,
                 hec_ras_hdf_file: str,
                 nodes_shp_file: str,
                 edges_shp_file: str,
                 feature_stats_file: str,
                 previous_timesteps: int = 0,
                 node_feat_config: dict[str, bool] = {},
                 edge_feat_config: dict[str, bool] = {},
                 normalize: bool = False,
                 trim_from_peak_water_depth: bool = False,
                 debug: bool = False,
                 logger: Logger = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        # Setup debug and logger
        log = print
        if logger is not None and hasattr(logger, 'log'):
            log = logger.log

        # Set file paths to load data from
        self.dataset_info_path = dataset_info_path
        self.hdf_file = hec_ras_hdf_file
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file
        self.feature_stats_file = feature_stats_file
        self.normalize = normalize

        # Set features to load
        self.node_feature_list = self._get_feature_list(node_feat_config)
        self.edge_feature_list = self._get_feature_list(edge_feat_config)

        # Initialize dataset variables
        self.previous_timesteps = previous_timesteps
        self.ts_from_peak_water_depth = 50 if trim_from_peak_water_depth else None
        self.dataset_info = {
            'node_features': [],
            'edge_features': [],
            'num_static_node_features': 0,
            'num_dynamic_node_features': 0,
            'num_static_edge_features': 0,
            'num_dynamic_edge_features': 0,
            'previous_timesteps': previous_timesteps,
            'is_normalized': normalize,
            'ts_from_peak_water_depth': self.ts_from_peak_water_depth,
        }
        self.feature_stats = {}
        self._enforce_dataset_consistency()

        super().__init__(root_dir, transform, pre_transform, pre_filter, log=debug)

        self.load(self.processed_paths[0])
        if len(self.feature_stats) == 0:
            self.feature_stats = self.get_feature_stats()

    @property
    def raw_file_names(self):
        return [self.hdf_file, self.nodes_shp_file, self.edges_shp_file]

    @property
    def processed_file_names(self):
        return ['complete_data.pt', self.feature_stats_file]

    def download(self):
        # Data must be downloaded manually and placed in the raw_dir
        pass

    def process(self):
        timesteps, edge_index = self._get_graph_properties()
        static_nodes = self._get_static_node_features()
        dynamic_nodes = self._get_dynamic_node_features()
        static_edges = self._get_static_edge_features()
        dynamic_edges = self._get_dynamic_edge_features()

        if self.ts_from_peak_water_depth is not None:
            # Trim the data from the peak water level
            peak_water_depth_ts = dynamic_nodes['water_depth'].sum(axis=1).argmax()
            last_ts = peak_water_depth_ts + self.ts_from_peak_water_depth
            timesteps = timesteps[:last_ts]
            dynamic_nodes = self._trim_features_from_peak_water_depth(dynamic_nodes, last_ts)
            dynamic_edges = self._trim_features_from_peak_water_depth(dynamic_edges, last_ts)

        dataset = []
        for i in range(len(timesteps) - 1):
            node_features = self._get_timestep_data(self.dataset_info['node_features'], static_nodes, dynamic_nodes, i)
            edge_features = self._get_timestep_data(self.dataset_info['edge_features'], static_edges, dynamic_edges, i)
            label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, i)

            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                        y=label_nodes, y_edge=label_edges, timestep=timesteps[i])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            dataset.append(data)

        self.save(dataset, self.processed_paths[0])

        self.save_dataset_info()
        self.save_feature_stats()

    def get_dataset_info(self) -> Dict | None:
        if not os.path.exists(self.dataset_info_path):
            return None

        dataset_info = file_utils.read_yaml_file(self.dataset_info_path)
        return dataset_info

    def save_dataset_info(self):
        if os.path.exists(self.dataset_info_path):
            dataset_info = file_utils.read_yaml_file(self.dataset_info_path)
            if self.root not in dataset_info['included_datasets']:
                dataset_info['included_datasets'].append(self.root)
        else:
            dataset_info = {**self.dataset_info, 'included_datasets': [self.root]}
        file_utils.save_to_yaml_file(self.dataset_info_path, dataset_info)

    def get_feature_stats(self) -> Dict:
        if not os.path.exists(self.processed_paths[1]):
            return {}

        feature_stats = file_utils.read_yaml_file(self.processed_paths[1])
        return feature_stats

    def save_feature_stats(self):
        file_utils.save_to_yaml_file(self.processed_paths[1], self.feature_stats)

    # =========== Helper Methods ===========

    def _get_feature_list(self, feature: dict[str, bool]) -> List[str]:
        return [name for name, is_included in feature.items() if is_included]

    def _enforce_dataset_consistency(self):
        dataset_info = self.get_dataset_info()
        if dataset_info is None:
            return

        if (set(dataset_info['node_features']) != set(self.node_feature_list)
            or set(dataset_info['edge_features']) != set(self.edge_feature_list)
            or dataset_info['previous_timesteps'] != self.dataset_info['previous_timesteps']
            or dataset_info['is_normalized'] != self.dataset_info['is_normalized']
            or dataset_info['ts_from_peak_water_depth'] != self.dataset_info['ts_from_peak_water_depth']):
            raise ValueError(f'Dataset features are inconsistent with previously loaded datasets. See {self.dataset_info_path}')

    def _get_graph_properties(self) -> Tuple[List, torch.Tensor]:
        timesteps = get_event_timesteps(self.raw_paths[0])
        edge_index = get_edge_index(self.raw_paths[2])
        edge_index = torch.from_numpy(edge_index)

        if self.debug:
            self.debug_helper.print_timesteps_info(timesteps)
            self.debug_helper.print_graph_properties(edge_index)

        return timesteps, edge_index

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

        if self.debug and timestep_idx == 0:
            feature_assignment = 'node' if feature_list == self.dataset_info['node_features'] else 'edge'
            self.debug_helper.print_data_format(feature_assignment, ordered_static_feature_list, ordered_dynamic_feature_list, self.previous_timesteps)

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
        label_nodes = node_dynamic_features['water_depth'][timestep_idx+1]
        label_nodes = label_nodes[:, None] # Reshape to [num_nodes, 1]
        label_nodes = torch.from_numpy(label_nodes)

        label_edges = edge_dynamic_features['velocity'][timestep_idx+1]
        label_edges = label_edges[:, None] # Reshape to [num_edges, 1]
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges

    # =========== Feature Retrieval Methods ===========

    def _get_static_node_features(self) -> Dict[str, np.ndarray]:
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(self.raw_paths[0]),
            "roughness": lambda: get_roughness(self.raw_paths[0]),
            "elevation": lambda: get_cell_elevation(self.raw_paths[1]),
        }

        return self._get_features(feature_list=self.node_feature_list,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP,
                                  feature_assignment='node',
                                  feature_type='static')

    def _get_dynamic_node_features(self) -> Dict[str, np.ndarray]:
        def get_water_depth():
            """Get water depth from water level and elevation"""
            water_level = get_water_level(self.raw_paths[0])
            elevation = get_cell_elevation(self.raw_paths[1])[None, :]
            water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
            return water_depth

        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "rainfall": lambda: get_rainfall(self.raw_paths[0]),
            "water_depth": get_water_depth,
        }

        return self._get_features(feature_list=self.node_feature_list,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP,
                                  feature_assignment='node',
                                  feature_type='dynamic')

    def _get_static_edge_features(self) -> Dict[str, np.ndarray]:
        STATIC_EDGE_RETRIEVAL_MAP = {
            "direction_x": lambda: get_edge_direction_x(self.raw_paths[0]),
            "direction_y": lambda: get_edge_direction_y(self.raw_paths[0]),
            "face_length": lambda: get_face_length(self.raw_paths[0]),
            "length": lambda: get_edge_length(self.raw_paths[2]),
            "slope": lambda: get_edge_slope(self.raw_paths[2]),
        }

        return self._get_features(feature_list=self.edge_feature_list,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP,
                                  feature_assignment='edge',
                                  feature_type='static')

    def _get_dynamic_edge_features(self) -> Dict[str, np.ndarray]:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: get_velocity(self.raw_paths[0]),
        }

        return self._get_features(feature_list=self.edge_feature_list,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP,
                                  feature_assignment='edge',
                                  feature_type='dynamic')

    def _get_features(self,
                      feature_list: List[str],
                      feature_retrieval_map: Dict[str, Callable],
                      feature_assignment: Literal['node', 'edge'],
                      feature_type: Literal['static', 'dynamic']) -> Dict[str, np.ndarray]:
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
            self.dataset_info[f'num_{feature_type}_{feature_assignment}_features'] += 1
            self.dataset_info[f'{feature_assignment}_features'].append(feature)

        if self.debug:
            self.debug_helper.print_loaded_features(feature_assignment, feature_type, features)

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
