import os
import gc
import torch
import numpy as np
import psutil

from numpy import ndarray
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List

from .flood_event_dataset import FloodEventDataset

class InMemoryFloodDataset(FloodEventDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = self.load_to_memory()

    def load_to_memory(self) -> List[Data]:
        # Load constant data
        constant_values = np.load(self.processed_paths[3])
        edge_index: ndarray = constant_values['edge_index']
        static_nodes: ndarray = constant_values['static_nodes']
        static_edges: ndarray = constant_values['static_edges']

        t_edge_index = torch.from_numpy(edge_index.copy())

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
                dynamic_values_path = self.processed_paths[event_idx + 4]
                dynamic_values = np.load(dynamic_values_path, allow_pickle=True)
                event_timesteps: ndarray = dynamic_values['event_timesteps']
                dynamic_nodes: ndarray = dynamic_values['dynamic_nodes']
                dynamic_edges: ndarray = dynamic_values['dynamic_edges']

                # Load physics-informed loss information
                if self.with_global_mass_loss:
                    edge_face_flow_per_ts: ndarray = dynamic_values['edge_face_flow_per_ts']
                    total_rainfall_per_ts: ndarray = dynamic_values['total_rainfall_per_ts']

                if self.with_local_mass_loss:
                    node_rainfall_per_ts: ndarray = dynamic_values['node_rainfall_per_ts']
                    edge_face_flow_per_ts: ndarray = dynamic_values['edge_face_flow_per_ts']

                curr_event_idx = event_idx

            # Create Data object for timestep
            within_event_idx = idx - start_idx + self.previous_timesteps # First timestep starts at self.previous_timesteps
            timestep = event_timesteps[within_event_idx]
            node_features = self._get_node_timestep_data(static_nodes, dynamic_nodes, within_event_idx)
            edge_features = self._get_edge_timestep_data(static_edges, dynamic_edges, edge_index, within_event_idx)
            label_nodes, label_edges = self._get_timestep_labels(dynamic_nodes, dynamic_edges, within_event_idx)

            global_mass_info = None
            if self.with_global_mass_loss:
                global_mass_info = self._get_global_mass_info_for_timestep(total_rainfall_per_ts,
                                                                           edge_face_flow_per_ts,
                                                                           within_event_idx)

            local_mass_info = None
            if self.with_local_mass_loss:
                local_mass_info = self._get_local_mass_info_for_timestep(node_rainfall_per_ts,
                                                                         edge_face_flow_per_ts,
                                                                         within_event_idx)

            data = Data(x=node_features,
                    edge_index=t_edge_index,
                    edge_attr=edge_features,
                    y=label_nodes,
                    y_edge=label_edges,
                    timestep=timestep,
                    global_mass_info=global_mass_info,
                    local_mass_info=local_mass_info)

            data_list.append(data)

        gc.collect()

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        self.log_func(f"RAM usage after loading dataset: {(memory_usage / (1024 ** 3)):.2f} GB")

        return data_list

    def get(self, idx):
        return self.data_list[idx]
