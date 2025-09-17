from data import FloodEventDataset
from torch import Tensor
from typing import Optional

# =============== Combined Functions ===============

def get_physics_info_node_edge(x: Tensor, edge_attr: Tensor, previous_timesteps: int, batch):
    curr_water_volume = get_curr_volume_from_node_features(x, previous_timesteps)
    curr_face_flow = get_curr_flow_from_edge_features(edge_attr, previous_timesteps)
    # Need to overwrite boundary conditions as these are masked
    curr_face_flow = overwrite_outflow_boundary(curr_face_flow, batch)
    return curr_water_volume, curr_face_flow

def get_physics_info_edge(edge_attr: Tensor, previous_timesteps: int, batch):
    curr_face_flow = get_curr_flow_from_edge_features(edge_attr, previous_timesteps)
    # Need to overwrite boundary conditions as these are masked
    curr_face_flow = overwrite_outflow_boundary(curr_face_flow, batch)
    return curr_face_flow

# =============== Individual Functions ===============

def get_curr_volume_from_node_features(x: Tensor, previous_timesteps: int) -> Tensor:
    water_volume_dyn_num = FloodEventDataset.DYNAMIC_NODE_FEATURES.index('water_volume') + 1
    num_static_node_features = len(FloodEventDataset.STATIC_NODE_FEATURES)
    curr_water_volume_idx = num_static_node_features + ((previous_timesteps + 1) * water_volume_dyn_num) - 1
    curr_water_volume = x[:, [curr_water_volume_idx]]
    return curr_water_volume

def get_curr_flow_from_edge_features(edge_attr: Tensor, previous_timesteps: int) -> Tensor:
    flow_dyn_num = FloodEventDataset.DYNAMIC_EDGE_FEATURES.index('face_flow') + 1
    num_static_edge_features = len(FloodEventDataset.STATIC_EDGE_FEATURES)
    curr_flow_idx = num_static_edge_features + ((previous_timesteps + 1) * flow_dyn_num) - 1
    curr_flow = edge_attr[:, [curr_flow_idx]]
    return curr_flow

def overwrite_outflow_boundary(edge_pred: Tensor, batch) -> Tensor:
    assert hasattr(batch, 'global_mass_info') or hasattr(batch, 'local_mass_info'), "Physics-informed data must be included in the dataset"
    physics_obj = batch.global_mass_info if hasattr(batch, 'global_mass_info') else batch.local_mass_info
    boundary_outflow = physics_obj['boundary_outflow']
    outflow_edges_mask = physics_obj['outflow_edges_mask']
    edge_pred[outflow_edges_mask] = boundary_outflow
    return edge_pred

def get_total_rainfall(batch, current_timestep: Optional[int] = None):
    assert hasattr(batch, 'global_mass_info'), "Global mass conservation data must be included in the dataset"
    total_rainfall = batch.global_mass_info['total_rainfall']
    if current_timestep is None:
        return total_rainfall
    assert len(total_rainfall.shape) == 2, "Current timestep can only be specified for per-timestep rainfall data from autoregressive datasets"
    return total_rainfall[:, current_timestep]

def get_rainfall(batch, current_timestep: Optional[int] = None):
    assert hasattr(batch, 'local_mass_info'), "Local mass conservation data must be included in the dataset"
    rainfall = batch.local_mass_info['rainfall']
    if current_timestep is None:
        return rainfall
    assert len(rainfall.shape) == 2, "Current timestep can only be specified for per-timestep rainfall data from autoregressive datasets"
    return rainfall[:, current_timestep]
