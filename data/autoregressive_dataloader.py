import numpy as np

from collections.abc import Iterator
from itertools import islice
from torch.utils.data import BatchSampler, SequentialSampler
from torch_geometric.loader import DataLoader

from .flood_event_dataset import FloodEventDataset

class AutoRegressiveBatchSampler(BatchSampler):
    def __init__(self, dataset: FloodEventDataset, batch_size: int, num_timesteps: int = 1):
        super().__init__(sampler=SequentialSampler(dataset), batch_size=batch_size, drop_last=False)

        self.num_timesteps = num_timesteps
        self.event_end_idx = [*dataset.event_start_idx[1:], dataset.total_rollout_timesteps]
        self.group_size = self.batch_size * self.num_timesteps
        min_event_size = np.min(np.diff(self.event_end_idx, prepend=0))
        assert min_event_size > self.group_size, "Number of datapoints for each event must be greater than the group size (batch size * num timesteps) for autoregressive training."

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)

        for event_end in self.event_end_idx:
            event_iter = islice(sampler_iter, event_end)

            group = [*islice(event_iter, self.group_size)]
            while group:
                num_complete = len(group) // self.num_timesteps
                inc = 0
                while inc < self.num_timesteps:
                    batch = group[inc::self.num_timesteps]
                    batch = batch[:num_complete]  # Ensure all batches are of equal size
                    yield batch
                    inc += 1

                group = [*islice(event_iter, self.group_size)]

class AutoRegressiveDataLoader(DataLoader):
    def __init__(self, dataset: FloodEventDataset, batch_size: int = 1, num_timesteps: int = 1, **kwargs):
        assert batch_size > num_timesteps, "Batch size must be greater than number of timesteps for autoregressive data loading."

        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', False)

        batch_sampler = AutoRegressiveBatchSampler(dataset, batch_size, num_timesteps)

        super().__init__(dataset, shuffle=False, batch_sampler=batch_sampler, **kwargs)
