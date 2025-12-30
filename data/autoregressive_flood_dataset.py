import torch
import numpy as np

from numpy import ndarray
from torch import Tensor
from typing import Tuple
from typing import Dict

from .flood_event_dataset import FloodEventDataset

from .hecras_data_retrieval import get_event_timesteps, get_water_volume

class AutoregressiveFloodDataset(FloodEventDataset):
    def __init__(self,
                 num_label_timesteps: int = 1,
                 *args, **kwargs):
        self.num_label_timesteps = num_label_timesteps
        super().__init__(*args, **kwargs)

    # =========== process() methods ===========

    def _set_event_properties(self):
        self._event_peak_idx = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        event_rollout_trim_start = self.previous_timesteps  # First timestep starts at self.previous_timesteps
        event_rollout_trim_end = self.num_label_timesteps # Trim the last timesteps depending on the number of label timesteps
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

            event_total_rollout_ts = trim_num_timesteps - event_rollout_trim_start - event_rollout_trim_end
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        assert len(self._event_peak_idx) == len(self.event_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self.event_start_idx) == len(self.event_run_ids), 'Mismatch in number of events and start indices.'

    # =========== get() methods ===========

    def _get_node_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        '''For edge autoregressive training'''
        ts_data = []
        end_ts = timestep_idx + self.num_label_timesteps
        # Get node features for each timestep in the label horizon
        for ts_idx in range(timestep_idx, end_ts):
            if ts_idx >= dynamic_features.shape[0]:
                raise IndexError(f'Timestep index {ts_idx} out of range for dynamic features with shape {dynamic_features.shape}.')

            ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_NODE_FEATURES, ts_idx)
            ts_features = self._get_timestep_features(static_features, ts_dynamic_features)
            ts_data.append(ts_features)

        ts_data = torch.stack(ts_data, dim=-1)  # (num_nodes, num_features, num_label_timesteps)
        return ts_data

    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        '''For node autoregressive training'''
        ts_data = []
        end_ts = timestep_idx + self.num_label_timesteps
        # Get edge features for each timestep in the label horizon
        for ts_idx in range(timestep_idx, end_ts):
            if ts_idx >= dynamic_features.shape[0]:
                raise IndexError(f'Timestep index {ts_idx} out of range for dynamic features with shape {dynamic_features.shape}.')

            ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, ts_idx)
            ts_features = self._get_timestep_features(static_features, ts_dynamic_features)
            ts_data.append(ts_features)

        ts_data = torch.stack(ts_data, dim=-1)  # (num_edges, num_features, num_label_timesteps)
        return ts_data

    def _get_timestep_labels(self, node_dynamic_features: ndarray, edge_dynamic_features: ndarray, timestep_idx: int) -> Tuple[Tensor, Tensor]:
        start_label_idx = timestep_idx + 1  # Labels are at the next timestep
        end_label_idx = start_label_idx + self.num_label_timesteps

        label_nodes_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        current_nodes = node_dynamic_features[start_label_idx-1:end_label_idx-1, :, label_nodes_idx]
        next_nodes = node_dynamic_features[start_label_idx:end_label_idx, :, label_nodes_idx]
        label_nodes = next_nodes - current_nodes
        label_nodes = label_nodes.T[:, None, :] # (num_nodes, 1, num_label_timesteps)
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        current_edges = edge_dynamic_features[start_label_idx-1:end_label_idx-1, :, label_edges_idx]
        next_edges = edge_dynamic_features[start_label_idx:end_label_idx, :, label_edges_idx]
        label_edges = next_edges - current_edges
        label_edges = label_edges.T[:, None, :] # (num_edges, 1, num_label_timesteps)
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges

    def _get_global_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, event_idx: int, timestep_idx: int) -> Dict[str, Tensor]:
        end_idx = timestep_idx + self.num_label_timesteps
        event_bc = self.boundary_conditions[event_idx]
        non_boundary_nodes_mask = ~event_bc.boundary_nodes_mask
        total_rainfall = node_rainfall_per_ts[timestep_idx:end_idx, non_boundary_nodes_mask].sum(axis=1)[None, :]

        total_rainfall = torch.from_numpy(total_rainfall)
        inflow_edges_mask = torch.from_numpy(event_bc.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(event_bc.outflow_edges_mask)

        return {
            'total_rainfall': total_rainfall,
            'inflow_edges_mask': inflow_edges_mask,
            'outflow_edges_mask': outflow_edges_mask,
        }

    def _get_local_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, event_idx: int, timestep_idx: int) -> Dict[str, Tensor]:
        end_ts = timestep_idx + self.num_label_timesteps
        rainfall = node_rainfall_per_ts[timestep_idx:end_ts].T

        rainfall = torch.from_numpy(rainfall)

        return {
            'rainfall': rainfall,
        }
