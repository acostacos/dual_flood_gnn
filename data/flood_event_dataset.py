import os
import torch
import numpy as np
import pandas as pd

from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Callable, Tuple, List, Dict, Optional
from utils.logger import Logger
from utils.file_utils import read_yaml_file, save_to_yaml_file

from .hecras_data_retrieval import get_event_timesteps, get_cell_area, get_min_cell_elevation, get_roughness,\
    get_rainfall, get_water_level, get_water_volume, get_edge_direction_x, get_edge_direction_y, \
    get_face_length, get_velocity, get_face_flow
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
                 inflow_boundary_edges: List[int] = [],
                 outflow_boundary_nodes: List[int] = [],
                 normalize: bool = True,
                 spin_up_timesteps: Optional[int] = None,
                 timesteps_from_peak: Optional[int] = None,
                 debug: bool = False,
                 logger: Logger = None,
                 force_reload: bool = False):
        self.log_func = print
        if logger is not None and hasattr(logger, 'log'):
            self.log_func = logger.log

        # File paths
        self.hec_ras_files, self.hec_ras_run_ids = self._get_hecras_files_from_summary(root_dir, dataset_summary_file)
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file

        # Dataset configurations
        self.previous_timesteps = previous_timesteps
        self.inflow_boundary_edges = inflow_boundary_edges
        self.outflow_boundary_nodes = outflow_boundary_nodes
        self.normalize = normalize
        self.spin_up_timesteps = spin_up_timesteps

        # Dataset variables
        self.num_static_node_features = len(FloodEventDataset.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(FloodEventDataset.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(FloodEventDataset.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(FloodEventDataset.DYNAMIC_EDGE_FEATURES)
        self.event_start_idx = []
        self.total_train_timesteps = 0
        self.feature_stats = {}

        super().__init__(root_dir, transform=None, pre_transform=None, pre_filter=None, log=debug, force_reload=force_reload)

        if len(self.feature_stats) == 0:
            self.feature_stats = self.load_feature_stats()
        if len(self.event_start_idx) == 0 or self.total_train_timesteps == 0:
            self.event_start_idx, self.total_train_timesteps = self.load_event_stats()

    @property
    def raw_file_names(self):
        return [self.nodes_shp_file, self.edges_shp_file, *self.hec_ras_files]

    @property
    def processed_file_names(self):
        dynamic_files = [f'dynamic_values_event_{run_id}.npz' for run_id in self.hec_ras_run_ids]
        return ['event_stats.yaml', 'features_stats.yaml', 'constant_values.npz', *dynamic_files]

    def download(self):
        # Data must be downloaded manually and placed in the raw dir
        pass

    def process(self):
        self.log_func('Processing Flood Event Dataset...')

        self.event_start_idx, self.total_train_timesteps, event_num_timesteps = self._get_event_properties()
        edge_index = self._get_edge_index()

        static_nodes = self._get_static_node_features()
        dynamic_nodes = self._get_dynamic_node_features()
        static_edges = self._get_static_edge_features()
        dynamic_edges = self._get_dynamic_edge_features()

        ghost_nodes = self._get_ghost_nodes()

        bc_info = self._create_boundary_conditions(ghost_nodes, edge_index, dynamic_nodes, dynamic_edges)
        new_boundary_nodes, new_boundary_edges, boundary_dynamic_nodes, boundary_dynamic_edges = bc_info

        # Delete ghost nodes
        static_nodes = np.delete(static_nodes, ghost_nodes, axis=0)
        dynamic_nodes = np.delete(dynamic_nodes, ghost_nodes, axis=1)

        boundary_edges_idx = np.any(np.isin(edge_index, ghost_nodes), axis=0).nonzero()[0]
        static_edges = np.delete(static_edges, boundary_edges_idx, axis=0)
        dynamic_edges = np.delete(dynamic_edges, boundary_edges_idx, axis=1)
        edge_index = np.delete(edge_index, boundary_edges_idx, axis=1)

        # Convert to undirected with flipped edge features
        edge_index, static_edges, dynamic_edges = self._to_undirected_flipped(edge_index, static_edges, dynamic_edges)

        np.savez_compressed(self.processed_paths[2],
                            edge_index=edge_index,
                            static_nodes=static_nodes,
                            static_edges=static_edges,
                            boundary_nodes=new_boundary_nodes,
                            boundary_edges=new_boundary_edges)
        self.log_func(f'Saved constant values to {self.processed_paths[2]}')

        start_idx = 0
        for i, num_ts in enumerate(event_num_timesteps):
            run_id = self.hec_ras_run_ids[i]
            end_idx = start_idx + num_ts

            event_dynamic_nodes = dynamic_nodes[start_idx:end_idx].copy()
            event_dynamic_edges = dynamic_edges[start_idx:end_idx].copy()
            event_bc_dynamic_nodes = boundary_dynamic_nodes[start_idx:end_idx].copy()
            event_bc_dynamic_edges = boundary_dynamic_edges[start_idx:end_idx].copy()

            save_path = self.processed_paths[i + 3]
            np.savez_compressed(save_path,
                                dynamic_nodes=event_dynamic_nodes,
                                dynamic_edges=event_dynamic_edges,
                                boundary_dynamic_nodes=event_bc_dynamic_nodes,
                                boundary_dynamic_edges=event_bc_dynamic_edges)
            self.log_func(f'Saved dynamic values for event {run_id} to {save_path}')

            start_idx = end_idx

        self.save_event_stats()
        self.log_func(f'Saved event stats to {self.processed_paths[0]}')
        self.save_feature_stats()
        self.log_func(f'Saved feature stats to {self.processed_paths[1]}')

    def len(self):
        return self.total_train_timesteps

    def get(self, idx):
        # Load constant data
        constant_values = np.load(self.processed_paths[2])
        edge_index = constant_values['edge_index']
        static_nodes = constant_values['static_nodes']
        static_edges = constant_values['static_edges']
        boundary_nodes = constant_values['boundary_nodes']
        boundary_edges = constant_values['boundary_edges']

        # Find the event this index belongs to using the start indices
        if idx < 0 or idx >= self.total_train_timesteps:
            raise IndexError(f'Index {idx} out of bounds for dataset with {self.total_train_timesteps} timesteps.')
        start_idx = 0
        for si in self.event_start_idx:
            if idx < si:
                break
            start_idx = si
        event_idx = self.event_start_idx.index(start_idx)

        # Load dynamic data
        dynamic_values_path = self.processed_paths[event_idx + 3]
        dynamic_values = np.load(dynamic_values_path)
        dynamic_nodes = dynamic_values['dynamic_nodes']
        dynamic_edges = dynamic_values['dynamic_edges']
        boundary_dynamic_nodes = dynamic_values['boundary_dynamic_nodes']
        boundary_dynamic_edges = dynamic_values['boundary_dynamic_edges']

        # Add boundary conditions
        boundary_static_nodes = np.zeros((len(boundary_nodes), self.num_static_node_features),
                                         dtype=static_nodes.dtype)
        static_nodes = np.concat([static_nodes, boundary_static_nodes], axis=0)

        boundary_static_edges = np.zeros((boundary_edges.shape[1], self.num_static_edge_features),
                                         dtype=static_edges.dtype)
        static_edges = np.concat([static_edges, boundary_static_edges], axis=0)

        dynamic_nodes = np.concat([dynamic_nodes, boundary_dynamic_nodes], axis=1)
        dynamic_edges = np.concat([dynamic_edges, boundary_dynamic_edges], axis=1)

        edge_index = np.concat([edge_index, boundary_edges], axis=1)

        # Create Data object for timestep
        edge_index = torch.from_numpy(edge_index)
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
    
    def _get_event_properties(self) -> Tuple[List[int], int, List[int]]:
        event_start_idx = []
        event_num_timesteps = []
        current_total_ts = 0
        for hec_ras_path in self.raw_paths[2:]:
            timesteps = get_event_timesteps(hec_ras_path)
            if self.spin_up_timesteps is not None:
                timesteps = timesteps[self.spin_up_timesteps:]
            num_timesteps = len(timesteps)
            event_num_timesteps.append(num_timesteps)

            event_total_training_ts = num_timesteps - 1  # Last timestep is used for labels
            event_start_idx.append(current_total_ts)
            current_total_ts += event_total_training_ts
        assert len(event_start_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and start indices.'

        return event_start_idx, current_total_ts, event_num_timesteps

    def _get_edge_index(self) -> ndarray:
        edge_index = get_edge_index(self.raw_paths[1])
        return edge_index

    def _get_ghost_nodes(self) -> ndarray:
        min_elevation = get_min_cell_elevation(self.raw_paths[2])
        ghost_nodes = np.where(np.isnan(min_elevation))[0]
        return ghost_nodes

    def _create_boundary_conditions(self,
                                    ghost_nodes: ndarray,
                                    edge_index: ndarray,
                                    dynamic_nodes: ndarray,
                                    dynamic_edges: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        inflow_boundary_nodes = np.unique(edge_index[:, self.inflow_boundary_edges])
        inflow_boundary_nodes = inflow_boundary_nodes[np.isin(inflow_boundary_nodes, ghost_nodes)]
        boundary_nodes = np.concat([inflow_boundary_nodes, np.array(self.outflow_boundary_nodes)])

        boundary_edges_mask = np.any(np.isin(edge_index, boundary_nodes), axis=0)
        boundary_edges = edge_index[:, boundary_edges_mask]
        boundary_edges_idx = boundary_edges_mask.nonzero()[0]

        # Reassign new indices to the boundary nodes taking into account the removal of ghost nodes
        # Ghost nodes are assumed to be the last nodes in the node feature matrix
        new_boundary_nodes = np.arange(ghost_nodes[0], (ghost_nodes[0] + len(boundary_nodes)))
        new_boundary_edges = boundary_edges.copy()
        boundary_nodes_mapping = dict(zip(boundary_nodes, new_boundary_nodes))
        for old_value, new_value in boundary_nodes_mapping.items():
            new_boundary_edges[new_boundary_edges == old_value] = new_value

        # Node boundary conditions = Outflow Water Volume
        outflow_dynamic_nodes = dynamic_nodes[:, self.outflow_boundary_nodes, :].copy()
        num_ts, _, num_dynamic_node_feat = dynamic_nodes.shape
        num_boundary_nodes = len(new_boundary_nodes)
        boundary_dynamic_nodes = np.zeros((num_ts, num_boundary_nodes, num_dynamic_node_feat), dtype=dynamic_nodes.dtype)

        target_nodes_idx = FloodEventDataset.DYNAMIC_NODE_FEATURES.index(FloodEventDataset.NODE_TARGET_FEATURE)
        outflow_dynamic_nodes_mask = np.isin(boundary_nodes, self.outflow_boundary_nodes)
        boundary_dynamic_nodes[:, outflow_dynamic_nodes_mask, target_nodes_idx] = outflow_dynamic_nodes[:, :, target_nodes_idx]

        # Edge boundary conditions = Inflow Water Flow
        inflow_dynamic_edges = dynamic_edges[:, self.inflow_boundary_edges, :].copy()
        num_ts, _, num_dynamic_edge_feat = dynamic_edges.shape
        num_boundary_edges = len(boundary_edges_idx)
        boundary_dynamic_edges = np.zeros((num_ts, num_boundary_edges, num_dynamic_edge_feat), dtype=dynamic_edges.dtype)

        target_edges_idx = FloodEventDataset.DYNAMIC_EDGE_FEATURES.index(FloodEventDataset.EDGE_TARGET_FEATURE)
        inflow_dynamic_edges_mask = np.isin(boundary_edges_idx, self.inflow_boundary_edges)
        boundary_dynamic_edges[:, inflow_dynamic_edges_mask, target_edges_idx] = inflow_dynamic_edges[:, :, target_edges_idx]

        # Ensure boundary edges are pointing away from the ghost nodes
        to_boundary = np.isin(new_boundary_edges[1], new_boundary_nodes)
        flipped_to_boundary = new_boundary_edges[:, to_boundary]
        flipped_to_boundary[[0, 1], :] = flipped_to_boundary[[1, 0], :]
        new_boundary_edges = np.concat([new_boundary_edges[:, ~to_boundary], flipped_to_boundary], axis=1)
        # Flip the dynamic edge features accordingly
        boundary_dynamic_edges[:, to_boundary, :] *= -1

        return new_boundary_nodes, new_boundary_edges, boundary_dynamic_nodes, boundary_dynamic_edges

    def _to_undirected_flipped(self, edge_index: ndarray, static_edges: ndarray, dynamic_edges: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        # Convert to undirected with flipped edge features
        row, col = edge_index[0], edge_index[1]
        row, col = np.concat([row, col], axis=0), np.concat([col, row], axis=0)
        edge_index = np.stack([row, col], axis=0)

        static_edges = np.concat([static_edges, static_edges], axis=0)
        flipped_dynamic_edges = dynamic_edges * -1
        dynamic_edges = np.concat([dynamic_edges, flipped_dynamic_edges], axis=1)

        return edge_index, static_edges, dynamic_edges

    def _get_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, previous_dynamic_features, current_dynamic_features]"""
        _, num_elems, num_dyn_features = dynamic_features.shape
        if timestep_idx < self.previous_timesteps:
            # Pad with zeros if not enough previous timesteps are available
            padding = np.zeros((self.previous_timesteps - timestep_idx, num_elems, num_dyn_features), dtype=dynamic_features.dtype)
            ts_dynamic_features = np.concat([padding, dynamic_features[:timestep_idx+1, :, :]], axis=0)
        else:
            ts_dynamic_features = dynamic_features[timestep_idx-self.previous_timesteps:timestep_idx+1, :, :]

        # (num_elems, num_timesteps * num_dynamic_features)
        ts_dynamic_features = ts_dynamic_features.transpose(1, 0, 2).reshape(num_elems, -1)

        ts_data = np.concat([static_features, ts_dynamic_features], axis=1)
        return torch.from_numpy(ts_data)

    def _get_timestep_labels(self, node_dynamic_features: ndarray, edge_dynamic_features: ndarray, timestep_idx: int) -> Tuple[Tensor, Tensor]:
        # Target feature must be the last feature in the dynamic features
        label_nodes_idx = FloodEventDataset.DYNAMIC_NODE_FEATURES.index(FloodEventDataset.NODE_TARGET_FEATURE)
        # (num_nodes, 1)
        label_nodes = node_dynamic_features[timestep_idx+1, :, label_nodes_idx][:, None]
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = FloodEventDataset.DYNAMIC_EDGE_FEATURES.index(FloodEventDataset.EDGE_TARGET_FEATURE)
        # (num_nodes, 1)
        label_edges = edge_dynamic_features[timestep_idx+1, :, label_edges_idx][:, None]
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges

    # =========== Feature Retrieval Methods ===========

    def _get_static_node_features(self) -> ndarray:
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(self.raw_paths[2]),
            "roughness": lambda: get_roughness(self.raw_paths[2]),
            "elevation": lambda: get_cell_elevation(self.raw_paths[0]),
        }

        static_features = self._get_features(feature_list=FloodEventDataset.STATIC_NODE_FEATURES,
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

        static_features = self._get_features(feature_list=FloodEventDataset.STATIC_EDGE_FEATURES,
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

        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "rainfall": lambda: self._get_dynamic_from_all_events(get_rainfall),
            "water_depth": lambda: self._get_dynamic_from_all_events(get_water_depth),
            "water_volume": lambda: self._get_dynamic_from_all_events(get_water_volume),
        }

        dynamic_features = self._get_features(feature_list=FloodEventDataset.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self) -> ndarray:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_events(get_velocity),
            "face_flow": lambda: self._get_dynamic_from_all_events(get_face_flow),
        }

        dynamic_features = self._get_features(feature_list=FloodEventDataset.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_from_all_events(self, retrieval_func: Callable) -> ndarray:
        all_event_data = []
        for hec_ras_path in self.raw_paths[2:]:
            event_data = retrieval_func(hec_ras_path)

            if self.spin_up_timesteps is not None:
                event_data = event_data[self.spin_up_timesteps:]

            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_features(self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]) -> List:
        features = []
        for feature in feature_list: # Order in feature list determines the order of features in the output
            if feature not in feature_retrieval_map:
                continue

            feature_data: ndarray = feature_retrieval_map[feature]()
            self.feature_stats[feature] = {
                'mean': feature_data.mean().item(),
                'std': feature_data.std().item(),
            }

            if self.normalize:
                feature_data = self._normalize_features(feature_data)

            features.append(feature_data)

        return features

    def _normalize_features(self, feature_data: ndarray) -> ndarray:
        """Z-score normalization of features"""
        EPS = 1e-7 # Prevent division by zero
        return (feature_data - feature_data.mean()) / (feature_data.std() + EPS)

    def _denormalize_features(self, feature: str, feature_data: ndarray) -> ndarray:
        """Z-score denormalization of features"""
        if feature not in self.feature_stats:
            raise ValueError(f'Feature {feature} not found in feature stats.')

        EPS = 1e-7
        return feature_data * (self.feature_stats[feature]['std'] + EPS) + self.feature_stats[feature]['mean']
