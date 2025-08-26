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
            peak_idx = np.argmax(total_water_volume)
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
