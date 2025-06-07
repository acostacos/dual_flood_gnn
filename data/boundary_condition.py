import numpy as np

from numpy import ndarray
from typing import List, Tuple

from .hecras_data_retrieval import get_min_cell_elevation

class BoundaryCondition:
    def __init__(self,
                 hec_ras_path: str,
                 inflow_boundary_edges: List[int],
                 outflow_boundary_nodes: List[int]):
        self.inflow_boundary_edges = inflow_boundary_edges
        self.outflow_boundary_nodes = outflow_boundary_nodes
        self.ghost_nodes = self._get_ghost_nodes(hec_ras_path)

        self.boundary_nodes = None
        self.boundary_edge_index = None
        self.boundary_dynamic_nodes = None
        self.boundary_dynamic_edges = None

    def apply(self,
              static_nodes: ndarray,
              dynamic_nodes: ndarray,
              static_edges: ndarray,
              dynamic_edges: ndarray,
              edge_index: ndarray) -> None:
        self._create_boundary_conditions(edge_index, dynamic_nodes, dynamic_edges)

        filtered_features = self._remove_ghost_nodes(static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index)
        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = filtered_features

        features_with_bc = self._add_boundary_conditions(static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index)
        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = features_with_bc

        return static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index

    def _get_ghost_nodes(self, hec_ras_path: str) -> ndarray:
        min_elevation = get_min_cell_elevation(hec_ras_path)
        ghost_nodes = np.where(np.isnan(min_elevation))[0]
        return ghost_nodes

    def _create_boundary_conditions(self,
                                    edge_index: ndarray,
                                    dynamic_nodes: ndarray,
                                    dynamic_edges: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        inflow_boundary_nodes = np.unique(edge_index[:, self.inflow_boundary_edges])
        inflow_boundary_nodes = inflow_boundary_nodes[np.isin(inflow_boundary_nodes, self.ghost_nodes)]
        boundary_nodes = np.concat([inflow_boundary_nodes, np.array(self.outflow_boundary_nodes)])

        for bn in boundary_nodes:
            assert bn in self.ghost_nodes, f'Boundary node {bn} is not a ghost node.'

        boundary_edges_mask = np.any(np.isin(edge_index, boundary_nodes), axis=0)
        boundary_edge_index = edge_index[:, boundary_edges_mask]
        boundary_edges = boundary_edges_mask.nonzero()[0]

        # Reassign new indices to the boundary nodes taking into account the removal of ghost nodes
        # Ghost nodes are assumed to be the last nodes in the node feature matrix
        _, num_nodes, _ = dynamic_nodes.shape
        num_non_ghost_nodes = num_nodes - len(self.ghost_nodes)
        new_boundary_nodes = np.arange(num_non_ghost_nodes, (num_non_ghost_nodes + len(boundary_nodes)))

        new_boundary_edge_index = boundary_edge_index.copy()
        boundary_nodes_mapping = dict(zip(boundary_nodes, new_boundary_nodes))
        for old_value, new_value in boundary_nodes_mapping.items():
            new_boundary_edge_index[new_boundary_edge_index == old_value] = new_value

        # Node boundary conditions = Outflow Water Volume
        # outflow_dynamic_nodes = dynamic_nodes[:, self.outflow_boundary_nodes, :].copy()
        # num_ts, _, num_dynamic_node_feat = dynamic_nodes.shape
        # num_boundary_nodes = len(new_boundary_nodes)
        # boundary_dynamic_nodes = np.zeros((num_ts, num_boundary_nodes, num_dynamic_node_feat), dtype=dynamic_nodes.dtype)

        # target_nodes_idx = FloodEventDataset.DYNAMIC_NODE_FEATURES.index(FloodEventDataset.NODE_TARGET_FEATURE)
        # outflow_dynamic_nodes_mask = np.isin(boundary_nodes, self.outflow_boundary_nodes)
        # boundary_dynamic_nodes[:, outflow_dynamic_nodes_mask, target_nodes_idx] = outflow_dynamic_nodes[:, :, target_nodes_idx]
        boundary_dynamic_nodes = dynamic_nodes[:, boundary_nodes, :].copy()

        # Edge boundary conditions = Inflow Water Flow
        # inflow_dynamic_edges = dynamic_edges[:, self.inflow_boundary_edges, :].copy()
        # num_ts, _, num_dynamic_edge_feat = dynamic_edges.shape
        # num_boundary_edges = len(boundary_edges)
        # boundary_dynamic_edges = np.zeros((num_ts, num_boundary_edges, num_dynamic_edge_feat), dtype=dynamic_edges.dtype)

        # target_edges_idx = FloodEventDataset.DYNAMIC_EDGE_FEATURES.index(FloodEventDataset.EDGE_TARGET_FEATURE)
        # inflow_dynamic_edges_mask = np.isin(boundary_edges, self.inflow_boundary_edges)
        # boundary_dynamic_edges[:, inflow_dynamic_edges_mask, target_edges_idx] = inflow_dynamic_edges[:, :, target_edges_idx]
        boundary_dynamic_edges = dynamic_edges[:, boundary_edges, :].copy()

        # Ensure inflow boundary edges are pointing away from the boundary nodes
        new_inflow_boundary_nodes = [new for old, new in boundary_nodes_mapping.items() if old in inflow_boundary_nodes]
        to_boundary = np.isin(new_boundary_edge_index[1], new_inflow_boundary_nodes)
        flipped_to_boundary = new_boundary_edge_index[:, to_boundary]
        flipped_to_boundary[[0, 1], :] = flipped_to_boundary[[1, 0], :]
        new_boundary_edge_index = np.concat([new_boundary_edge_index[:, ~to_boundary], flipped_to_boundary], axis=1)
        # Flip the dynamic edge features accordingly
        boundary_dynamic_edges[:, to_boundary, :] *= -1

        # # Ensure boundary edges are pointing away from the ghost nodes
        # to_boundary = np.isin(new_boundary_edge_index[1], new_boundary_nodes)
        # flipped_to_boundary = new_boundary_edge_index[:, to_boundary]
        # flipped_to_boundary[[0, 1], :] = flipped_to_boundary[[1, 0], :]
        # new_boundary_edge_index = np.concat([new_boundary_edge_index[:, ~to_boundary], flipped_to_boundary], axis=1)
        # # Flip the dynamic edge features accordingly
        # boundary_dynamic_edges[:, to_boundary, :] *= -1

        self.boundary_nodes = new_boundary_nodes
        self.boundary_edge_index = new_boundary_edge_index
        self.boundary_dynamic_nodes = boundary_dynamic_nodes
        self.boundary_dynamic_edges = boundary_dynamic_edges

    def _remove_ghost_nodes(self,
                           static_nodes: ndarray,
                           dynamic_nodes: ndarray,
                           static_edges: ndarray,
                           dynamic_edges: ndarray,
                           edge_index: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        static_nodes = np.delete(static_nodes, self.ghost_nodes, axis=0)
        dynamic_nodes = np.delete(dynamic_nodes, self.ghost_nodes, axis=1)

        ghost_edges_idx = np.any(np.isin(edge_index, self.ghost_nodes), axis=0).nonzero()[0]
        static_edges = np.delete(static_edges, ghost_edges_idx, axis=0)
        dynamic_edges = np.delete(dynamic_edges, ghost_edges_idx, axis=1)
        edge_index = np.delete(edge_index, ghost_edges_idx, axis=1)
        return static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index

    def _add_boundary_conditions(self,
                                 static_nodes: ndarray,
                                 dynamic_nodes: ndarray,
                                 static_edges: ndarray,
                                 dynamic_edges: ndarray,
                                 edge_index: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        _, num_static_node_feat = static_nodes.shape
        boundary_static_nodes = np.zeros((len(self.boundary_nodes), num_static_node_feat),
                                        dtype=static_nodes.dtype)
        static_nodes = np.concat([static_nodes, boundary_static_nodes], axis=0)

        _, num_static_edge_feat = static_edges.shape
        boundary_static_edges = np.zeros((self.boundary_edge_index.shape[1], num_static_edge_feat),
                                         dtype=static_edges.dtype)
        static_edges = np.concat([static_edges, boundary_static_edges], axis=0)

        dynamic_nodes = np.concat([dynamic_nodes, self.boundary_dynamic_nodes], axis=1)
        dynamic_edges = np.concat([dynamic_edges, self.boundary_dynamic_edges], axis=1)

        edge_index = np.concat([edge_index, self.boundary_edge_index], axis=1)

        return static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index
