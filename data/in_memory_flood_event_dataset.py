import gc
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data
from typing import List

from .flood_event_dataset import FloodEventDataset

class InMemoryFloodEventDataset(FloodEventDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = self.load_to_memory()

    def load_to_memory(self) -> List[Data]:
        # Load constant data
        constant_values = np.load(self.processed_paths[2])
        edge_index = constant_values['edge_index']
        static_nodes = constant_values['static_nodes']
        static_edges = constant_values['static_edges']
        boundary_nodes = constant_values['boundary_nodes']
        boundary_edges = constant_values['boundary_edges']

        # Add static boundary conditions
        boundary_static_nodes = self._get_normalized_zeros_for_features(FloodEventDataset.STATIC_NODE_FEATURES,
                                                                        (len(boundary_nodes),),
                                                                        dtype=static_nodes.dtype)
        boundary_nodes_idx = np.arange(static_nodes.shape[0], static_nodes.shape[0] + len(boundary_nodes))
        static_nodes = np.concat([static_nodes, boundary_static_nodes], axis=0)

        boundary_static_edges = self._get_normalized_zeros_for_features(FloodEventDataset.STATIC_EDGE_FEATURES,
                                                                        (boundary_edges.shape[1],),
                                                                        dtype=static_edges.dtype)
        boundary_edges_idx = np.arange(static_edges.shape[0], static_edges.shape[0] + boundary_edges.shape[1])
        static_edges = np.concat([static_edges, boundary_static_edges], axis=0)

        edge_index = np.concat([edge_index, boundary_edges], axis=1)

        boundary_nodes_idx = torch.from_numpy(boundary_nodes_idx)
        boundary_edges_idx = torch.from_numpy(boundary_edges_idx)
        edge_index = torch.from_numpy(edge_index)

        curr_event_idx = -1
        data_list = []
        progress_bar = tqdm(range(self.total_rollout_timesteps), desc='Processing timesteps')
        for idx in progress_bar:
            # Find the event this index belongs to using the start indices
            if idx < 0 or idx >= self.total_rollout_timesteps:
                raise IndexError(f'Index {idx} out of bounds for dataset with {self.total_rollout_timesteps} timesteps.')
            start_idx = 0
            for si in self.event_start_idx:
                if idx < si:
                    break
                start_idx = si
            event_idx = self.event_start_idx.index(start_idx)

            if event_idx != curr_event_idx:
                # Load dynamic data
                dynamic_values_path = self.processed_paths[event_idx + 3]
                dynamic_values = np.load(dynamic_values_path)
                dynamic_nodes = dynamic_values['dynamic_nodes']
                dynamic_edges = dynamic_values['dynamic_edges']
                boundary_dynamic_nodes = dynamic_values['boundary_dynamic_nodes']
                boundary_dynamic_edges = dynamic_values['boundary_dynamic_edges']
                curr_event_idx = event_idx

                # Add dynamic boundary conditions
                dynamic_nodes = np.concat([dynamic_nodes, boundary_dynamic_nodes], axis=1)
                dynamic_edges = np.concat([dynamic_edges, boundary_dynamic_edges], axis=1)

            # Create Data object for timestep
            within_event_idx = idx - start_idx

            node_features = self._get_timestep_data(static_nodes, dynamic_nodes, FloodEventDataset.DYNAMIC_NODE_FEATURES, within_event_idx)
            edge_features = self._get_timestep_data(static_edges, dynamic_edges, FloodEventDataset.DYNAMIC_EDGE_FEATURES, within_event_idx)

            label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, within_event_idx)

            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=label_nodes,
                        y_edge=label_edges,
                        boundary_nodes=boundary_nodes_idx,
                        boundary_edges=boundary_edges_idx)
            data_list.append(data)

        gc.collect()

        return data_list

    def get(self, idx):
        return self.data_list[idx]
