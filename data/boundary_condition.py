import os
import numpy as np

from numpy import ndarray
from torch import Tensor
from typing import List, Tuple, Union

from .hecras_data_retrieval import get_min_cell_elevation

class BoundaryCondition:
    def __init__(self,
                 root_dir: str,
                 hec_ras_file: str,
                 inflow_boundary_nodes: List[int],
                 outflow_boundary_nodes: List[int]):
        self.hec_ras_path = os.path.join(root_dir, 'raw', hec_ras_file)
        self.init_inflow_boundary_nodes = inflow_boundary_nodes
        self.init_outflow_boundary_nodes = outflow_boundary_nodes
        self._init()

        self.outflow_edges_mask = None # Used for physics mass conservation loss
        self.non_boundary_edges_mask = None # Used for testing

        self._boundary_edge_index = None
        self._boundary_dynamic_nodes = None
        self._boundary_dynamic_edges = None

    def _init(self) -> None:
        min_elevation = get_min_cell_elevation(self.hec_ras_path)
        ghost_nodes = np.where(np.isnan(min_elevation))[0]

        boundary_nodes = np.concat([np.array(self.init_inflow_boundary_nodes), np.array(self.init_outflow_boundary_nodes)])
        for bn in boundary_nodes:
            assert bn in ghost_nodes, f'Boundary node {bn} is not a ghost node.'

        # Reassign new indices to the boundary nodes taking into account the removal of ghost nodes
        # Ghost nodes are assumed to be the last nodes in the node feature matrix
        num_nodes, = min_elevation.shape
        num_non_ghost_nodes = num_nodes - len(ghost_nodes)
        new_boundary_nodes = np.arange(num_non_ghost_nodes, (num_non_ghost_nodes + len(boundary_nodes)))
        boundary_nodes_mapping = dict(zip(boundary_nodes, new_boundary_nodes))
        new_inflow_boundary_nodes = np.array([boundary_nodes_mapping[bn] for bn in self.init_inflow_boundary_nodes])
        new_outflow_boundary_nodes = np.array([boundary_nodes_mapping[bn] for bn in self.init_outflow_boundary_nodes])

        self.ghost_nodes = ghost_nodes
        self.boundary_nodes_mapping = boundary_nodes_mapping
        self.new_num_nodes = num_non_ghost_nodes + len(boundary_nodes)
        self.new_inflow_boundary_nodes = new_inflow_boundary_nodes
        self.new_outflow_boundary_nodes = new_outflow_boundary_nodes

    def create(self, edge_index: ndarray, dynamic_nodes: ndarray, dynamic_edges: ndarray) -> None:
        boundary_nodes = np.concat([np.array(self.init_inflow_boundary_nodes), np.array(self.init_outflow_boundary_nodes)])

        boundary_edges_mask = np.any(np.isin(edge_index, boundary_nodes), axis=0)
        boundary_edge_index = edge_index[:, boundary_edges_mask]
        boundary_edges = boundary_edges_mask.nonzero()[0]

        new_boundary_edge_index = boundary_edge_index.copy()
        for old_value, new_value in self.boundary_nodes_mapping.items():
            new_boundary_edge_index[new_boundary_edge_index == old_value] = new_value

        boundary_dynamic_nodes = dynamic_nodes[:, boundary_nodes, :].copy()
        boundary_dynamic_edges = dynamic_edges[:, boundary_edges, :].copy()

        # Ensure inflow boundary edges point away from the boundary node
        inflow_to_boundary_mask = np.isin(new_boundary_edge_index[1], self.new_inflow_boundary_nodes)
        if np.any(inflow_to_boundary_mask):
            inflow_to_boundary = new_boundary_edge_index[:, inflow_to_boundary_mask]
            inflow_to_boundary[[0, 1], :] = inflow_to_boundary[[1, 0], :]
            new_boundary_edge_index[:, inflow_to_boundary_mask] = inflow_to_boundary
            # Flip the dynamic edge features accordingly
            boundary_dynamic_edges[:, inflow_to_boundary_mask, :] *= -1

        # Ensure outflow boundary edges point towards the boundary node
        outflow_from_boundary_mask = np.isin(new_boundary_edge_index[0], self.new_outflow_boundary_nodes)
        if np.any(outflow_from_boundary_mask):
            outflow_from_boundary = new_boundary_edge_index[:, outflow_from_boundary_mask]
            outflow_from_boundary[[0, 1], :] = outflow_from_boundary[[1, 0], :]
            new_boundary_edge_index[:, outflow_from_boundary_mask] = outflow_from_boundary
            # Flip the dynamic edge features accordingly
            boundary_dynamic_edges[:, outflow_from_boundary_mask, :] *= -1

        self._boundary_edge_index = new_boundary_edge_index
        self._boundary_dynamic_nodes = boundary_dynamic_nodes
        self._boundary_dynamic_edges = boundary_dynamic_edges

    def remove(self,
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

    def apply(self,
            static_nodes: ndarray,
            dynamic_nodes: ndarray,
            static_edges: ndarray,
            dynamic_edges: ndarray,
            edge_index: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        _, num_static_node_feat = static_nodes.shape
        num_boundary_nodes = len(self.new_inflow_boundary_nodes) + len(self.new_outflow_boundary_nodes)
        boundary_static_nodes = np.zeros((num_boundary_nodes, num_static_node_feat),
                                        dtype=static_nodes.dtype)
        static_nodes = np.concat([static_nodes, boundary_static_nodes], axis=0)

        _, num_static_edge_feat = static_edges.shape
        boundary_static_edges = np.zeros((self._boundary_edge_index.shape[1], num_static_edge_feat),
                                         dtype=static_edges.dtype)
        static_edges = np.concat([static_edges, boundary_static_edges], axis=0)

        dynamic_nodes = np.concat([dynamic_nodes, self._boundary_dynamic_nodes], axis=1)
        dynamic_edges = np.concat([dynamic_edges, self._boundary_dynamic_edges], axis=1)

        edge_index = np.concat([edge_index, self._boundary_edge_index], axis=1)

        # Clear boundary condition attributes to save memory
        self._boundary_dynamic_edges = None
        self._boundary_dynamic_nodes = None
        self._boundary_edge_index = None

        return static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index

    def get_new_boundary_nodes(self) -> ndarray:
        return np.union1d(self.new_inflow_boundary_nodes, self.new_outflow_boundary_nodes)

    def get_non_boundary_nodes_mask(self) -> Union[ndarray, Tensor]:
        boundary_nodes = self.get_new_boundary_nodes()
        non_boundary_nodes_mask = ~np.isin(np.arange(self.new_num_nodes), boundary_nodes)
        return non_boundary_nodes_mask
