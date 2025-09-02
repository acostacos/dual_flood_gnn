import torch
import numpy as np

from numpy import ndarray
from torch import Tensor
from typing import Tuple

from .flood_event_dataset import FloodEventDataset

from .hecras_data_retrieval import get_event_timesteps, get_water_volume

class AutoregressiveFloodDataset(FloodEventDataset):
    def __init__(self,
                 num_label_timesteps: int = 1,
                 *args, **kwargs):
        self.num_label_timesteps = num_label_timesteps
        super().__init__(*args, **kwargs)

    # =========== process() methods ===========

    def _set_event_properties(self) -> ndarray:
        self._event_peak_idx = []
        self._event_num_timesteps = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        event_rollout_trim_start = self.previous_timesteps  # First timestep starts at self.previous_timesteps
        event_rollout_trim_end = self.num_label_timesteps # Trim the last timesteps depending on the number of label timesteps
        current_total_ts = 0
        all_event_timesteps = []
        for event_idx, hec_ras_path in enumerate(self.raw_paths[2:]):
            timesteps = get_event_timesteps(hec_ras_path)
            event_ts_interval = int((timesteps[1] - timesteps[0]).total_seconds())
            assert self.timestep_interval % event_ts_interval == 0, f'Event {self.hec_ras_run_ids[event_idx]} has a timestep interval of {event_ts_interval} seconds, which is not compatible with the dataset timestep interval of {self.timestep_interval} seconds.'
            self._event_base_timestep_interval.append(event_ts_interval)

            water_volume = get_water_volume(hec_ras_path)
            total_water_volume = water_volume.sum(axis=1)
            peak_idx = np.argmax(total_water_volume).item()
            num_timesteps_after_peak = self.time_from_peak // event_ts_interval if self.time_from_peak is not None else 0
            assert peak_idx + num_timesteps_after_peak < len(timesteps), "Timesteps after peak exceeds the available timesteps."
            self._event_peak_idx.append(peak_idx)

            timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx)
            all_event_timesteps.append(timesteps)

            num_timesteps = len(timesteps)
            self._event_num_timesteps.append(num_timesteps)

            event_total_rollout_ts = num_timesteps - event_rollout_trim_start - event_rollout_trim_end
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        assert len(self._event_peak_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self._event_num_timesteps) == len(self.hec_ras_run_ids), 'Mismatch in number of events and number of timesteps.'
        assert len(self.event_start_idx) == len(self.hec_ras_run_ids), 'Mismatch in number of events and start indices.'

        all_event_timesteps = np.concatenate(all_event_timesteps, axis=0)
        return all_event_timesteps

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
            ts_features = self._get_timestep_features(static_features, ts_dynamic_features)
            ts_data.append(ts_features)

        ts_data = torch.stack(ts_data, dim=-1)  # (num_nodes, num_features, num_label_timesteps)
        return ts_data

    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, edge_index: ndarray, timestep_idx: int) -> Tensor:
        '''For node autoregressive training'''
        ts_data = []
        end_ts = timestep_idx + self.num_label_timesteps
        # Get edge features for each timestep in the label horizon
        for ts_idx in range(timestep_idx, end_ts):
            if ts_idx >= dynamic_features.shape[0]:
                raise IndexError(f'Timestep index {ts_idx} out of range for dynamic features with shape {dynamic_features.shape}.')

            ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, ts_idx)

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

            ts_features = self._get_timestep_features(static_features, ts_dynamic_features)
            ts_data.append(ts_features)

        ts_data = torch.stack(ts_data, dim=-1)  # (num_edges, num_features, num_label_timesteps)
        return ts_data

    def _get_timestep_labels(self, node_dynamic_features: ndarray, edge_dynamic_features: ndarray, timestep_idx: int) -> Tuple[Tensor, Tensor]:
        start_label_idx = timestep_idx + 1  # Labels are at the next timestep
        end_label_idx = start_label_idx + self.num_label_timesteps

        label_nodes_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        # (num_nodes, 1, num_label_timesteps)
        label_nodes = node_dynamic_features[start_label_idx:end_label_idx, :, label_nodes_idx]
        label_nodes = label_nodes.T[:, None, :]
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        # (num_nodes, 1, num_label_timesteps)
        label_edges = edge_dynamic_features[start_label_idx:end_label_idx, :, label_edges_idx]
        label_edges = label_edges.T[:, None, :]
        label_edges = torch.from_numpy(label_edges)

        return label_nodes, label_edges
