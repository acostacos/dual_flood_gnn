import torch

from torch_geometric.loader import DataLoader
from utils.validation_stats import ValidationStats

from .base_tester import BaseTester

class EdgeRegressionTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self):
        for event_idx, run_id in enumerate(self.dataset.event_run_ids):
            self.log(f'Validating on run {event_idx + 1}/{len(self.dataset.event_run_ids)} with Run ID {run_id}')

            validation_stats = ValidationStats(logger=self.logger,
                                                normalizer=self.dataset.normalizer,
                                                is_normalized=self.dataset.is_normalized,
                                                delta_t=self.dataset.timestep_interval)
            self.run_test_for_event(event_idx, validation_stats)
            validation_stats.print_stats_summary()
            self.events_validation_stats.append(validation_stats)

        self.log(f'Average Edge RMSE across events: {self.get_avg_edge_rmse():.4e}')

    def run_test_for_event(self, event_idx: int, validation_stats: ValidationStats):
        validation_stats.start_validate()
        self.model.eval()
        with torch.no_grad():
            event_start_idx = self.dataset.event_start_idx[event_idx] + self.rollout_start
            event_end_idx = self.dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(self.dataset.event_start_idx) else self.dataset.total_rollout_timesteps
            if self.rollout_timesteps is not None:
                event_end_idx = event_start_idx + self.rollout_timesteps
                assert event_end_idx <= (self.dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(self.dataset.event_start_idx) else self.dataset.total_rollout_timesteps), \
                    f'Event end index {event_end_idx} exceeds dataset length {self.dataset.total_rollout_timesteps} for event_idx {event_idx}.'
            event_dataset = self.dataset[event_start_idx:event_end_idx]
            dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False) # Enforce batch size = 1 for autoregressive testing

            for graph in dataloader:
                graph = graph.to(self.device)

                x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
                edge_pred_diff = self.model(x, edge_index, edge_attr)

                # Override boundary conditions in predictions
                edge_pred_diff[graph.boundary_edges_mask] = graph.y_edge[graph.boundary_edges_mask]

                edge_pred = edge_attr[:, [self.end_edge_target_idx-1]] + edge_pred_diff

                label_edge = edge_attr[:, [self.end_edge_target_idx-1]] + graph.y_edge
                if self.dataset.is_normalized:
                    edge_pred = self.dataset.normalizer.denormalize(self.dataset.EDGE_TARGET_FEATURE, edge_pred)
                    label_edge = self.dataset.normalizer.denormalize(self.dataset.EDGE_TARGET_FEATURE, label_edge)

                validation_stats.update_edge_stats_for_timestep(edge_pred.cpu(), label_edge.cpu())
        validation_stats.end_validate()
