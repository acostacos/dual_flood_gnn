import os
import numpy as np

from data.boundary_condition import BoundaryCondition

from .shp_data_retrieval import get_node_types

class mSWEGNNBoundaryCondition(BoundaryCondition):
    def __init__(self,
                 root_dir: str,
                 nodes_shp_file: str,
                 *args, **kwargs):
        self.nodes_shp_path = os.path.join(root_dir, 'raw', nodes_shp_file)
        super().__init__(root_dir=root_dir, *args, **kwargs)
    
    def _init(self) -> None:
        # Custom logic for identifying boundary and ghost nodes
        node_types = get_node_types(self.nodes_shp_path) # 1: normal node, 2: boundary node, 3: ghost node
        ghost_nodes = np.where(node_types != 1)[0]

        inflow_boundary_nodes = np.where(node_types == 2)[0]
        self.init_inflow_boundary_nodes = inflow_boundary_nodes
        self.init_outflow_boundary_nodes = []


        # Resume original initialization logic
        boundary_nodes = np.concat([np.array(self.init_inflow_boundary_nodes), np.array(self.init_outflow_boundary_nodes)])
        for bn in boundary_nodes:
            assert bn in ghost_nodes, f'Boundary node {bn} is not a ghost node.'

        # Reassign new indices to the boundary nodes taking into account the removal of ghost nodes
        # Ghost nodes are assumed to be the last nodes in the node feature matrix
        num_nodes, = node_types.shape
        num_non_ghost_nodes = num_nodes - len(ghost_nodes)
        new_boundary_nodes = np.arange(num_non_ghost_nodes, (num_non_ghost_nodes + len(boundary_nodes)))
        boundary_nodes_mapping = dict(zip(boundary_nodes, new_boundary_nodes))
        new_inflow_boundary_nodes = np.array([boundary_nodes_mapping[bn] for bn in self.init_inflow_boundary_nodes])
        new_outflow_boundary_nodes = np.array([boundary_nodes_mapping[bn] for bn in self.init_outflow_boundary_nodes])

        self.ghost_nodes = ghost_nodes
        self.boundary_nodes_mapping = boundary_nodes_mapping
        self.new_inflow_boundary_nodes = new_inflow_boundary_nodes
        self.new_outflow_boundary_nodes = new_outflow_boundary_nodes

