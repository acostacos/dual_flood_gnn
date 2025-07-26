import numpy as np

from collections.abc import Iterator
from itertools import islice
from torch.utils.data import BatchSampler, SequentialSampler
from torch_geometric.loader import DataLoader

from .flood_event_dataset import FloodEventDataset

class AutoRegressiveBatchSampler(BatchSampler):
    def __init__(self, dataset: FloodEventDataset, batch_size: int, num_timesteps: int = 1):
        super().__init__(sampler=SequentialSampler(dataset), batch_size=batch_size, drop_last=False)

        event_end_idx = [*dataset.event_start_idx[1:], dataset.total_rollout_timesteps]
        self.event_size = np.diff(event_end_idx, prepend=0)

        self.num_timesteps = num_timesteps
        self.group_size = self.batch_size * self.num_timesteps
        assert np.min(self.event_size) > self.group_size, "Number of datapoints for each event must be greater than the group size (batch size * num timesteps) for autoregressive training."

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)

        for num_event_ts in self.event_size:
            event_iter = islice(sampler_iter, num_event_ts)

            group = [*islice(event_iter, self.group_size)]
            while group:
                if len(group) < self.num_timesteps:
                    # Group does not have enough timesteps for autoregressive training so skip it
                    group = None
                    continue

                num_complete = len(group) // self.num_timesteps
                inc = 0
                while inc < self.num_timesteps:
                    batch = group[inc::self.num_timesteps]
                    batch = batch[:num_complete]  # Ensure all batches are of equal size
                    yield batch
                    inc += 1

                group = [*islice(event_iter, self.group_size)]

    def __len__(self) -> int:
        num_complete_batch = (self.event_size // self.group_size) * self.num_timesteps
        num_remaining_batch = np.array([self.num_timesteps] * len(self.event_size))
        num_remaining = self.event_size % self.group_size
        num_remaining_batch[num_remaining < self.num_timesteps] = 0 # Remove groups that do not have enough timesteps
        num_batch = num_complete_batch + num_remaining_batch
        return num_batch.sum().item()

class AutoRegressiveDataLoader(DataLoader):
    def __init__(self, dataset: FloodEventDataset, batch_size: int = 1, num_timesteps: int = 1, **kwargs):
        assert batch_size > num_timesteps, "Batch size must be greater than number of timesteps for autoregressive data loading."

        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', False)

        batch_sampler = AutoRegressiveBatchSampler(dataset, batch_size, num_timesteps)

        super().__init__(dataset, shuffle=False, batch_sampler=batch_sampler, **kwargs)
