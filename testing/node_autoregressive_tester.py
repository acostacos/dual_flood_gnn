import torch

from torch_geometric.loader import DataLoader
from utils.validation_stats import ValidationStats

from .base_tester import BaseTester

class NodeAutoregressiveTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self):
        for event_idx, run_id in enumerate(self.dataset.hec_ras_run_ids):
            self.log(f'Validating on run {event_idx + 1}/{len(self.dataset.hec_ras_run_ids)} with Run ID {run_id}')

            validation_stats = ValidationStats(logger=self.logger,
                                                previous_timesteps=self.dataset.previous_timesteps,
                                                normalizer=self.dataset.normalizer,
                                                is_normalized=self.dataset.is_normalized,
                                                delta_t=self.dataset.timestep_interval)
            self.run_test_for_event(event_idx, validation_stats)
            validation_stats.print_stats_summary()
            self.events_validation_stats.append(validation_stats)

        self.log(f'Average Node RMSE across events: {self.get_avg_node_rmse():.4e}')
        if self.include_physics_loss:
            self.log(f'Average Global Mass Conservation Loss across events: {self.get_avg_global_mass_loss():.4e}')
            self.log(f'Average Local Mass Conservation Loss across events: {self.get_avg_local_mass_loss():.4e}')

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

            sliding_window = self.dataset[event_start_idx].x.clone()[:, self.start_node_target_idx:self.end_node_target_idx]
            sliding_window = sliding_window.to(self.device)
            for graph in dataloader:
                graph = graph.to(self.device)

                # Override graph data with sliding window
                # Only override non-boundary nodes to keep boundary conditions intact
                graph.x[self.non_boundary_nodes_mask, self.start_node_target_idx:self.end_node_target_idx] = \
                    sliding_window[self.non_boundary_nodes_mask]

                pred = self.model(graph)

                # Override boundary conditions in predictions
                pred[self.boundary_nodes_mask] = graph.y[self.boundary_nodes_mask]

                sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)

                # Requires normalized physics-informed loss
                if self.include_physics_loss:
                    # Requires normalized prediction for physics-informed loss
                    prev_edge_pred = graph.global_mass_info['face_flow']
                    validation_stats.update_physics_informed_stats_for_timestep(pred, graph, prev_edge_pred)

                label = graph.y
                if self.dataset.is_normalized:
                    pred = self.dataset.normalizer.denormalize(self.dataset.NODE_TARGET_FEATURE, pred)
                    label = self.dataset.normalizer.denormalize(self.dataset.NODE_TARGET_FEATURE, label)

                # Ensure water volume is non-negative
                pred = torch.clip(pred, min=0)
                label = torch.clip(label, min=0)

                # Filter boundary conditions for metric computation
                pred = pred[self.non_boundary_nodes_mask]
                label = label[self.non_boundary_nodes_mask]

                validation_stats.update_stats_for_timestep(pred.cpu(),
                                                           label.cpu(),
                                                           water_threshold=self.threshold_per_cell,
                                                           timestamp=graph.timestep)
        validation_stats.end_validate()
