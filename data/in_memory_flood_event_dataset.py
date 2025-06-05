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

        boundary_nodes = torch.from_numpy(boundary_nodes)
        boundary_edges = torch.from_numpy(boundary_edges)
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
                curr_event_idx = event_idx

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
                        boundary_nodes=boundary_nodes,
                        boundary_edges=boundary_edges)
            data_list.append(data)

        gc.collect()

        return data_list

    def get(self, idx):
        return self.data_list[idx]
