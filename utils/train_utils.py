import os
import pandas as pd

from data import FloodEventDataset
from typing import Tuple
from torch import Tensor

from .logger import Logger
from .file_utils import create_temp_dirs
from .model_utils import get_loss_func

def split_dataset_events(root_dir: str, dataset_summary_file: str, percent_validation: float) -> Tuple[str, str]:
    if not (0 < percent_validation < 1):
        raise ValueError(f'Invalid percent_split: {percent_validation}. Must be between 0 and 1.')

    raw_dir_path = os.path.join(root_dir, 'raw')
    dataset_summary_path = os.path.join(raw_dir_path, dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

    split_idx = len(summary_df) - int(len(summary_df) * percent_validation)

    TEMP_DIR_NAME = 'train_val_split'
    create_temp_dirs(raw_dir_path, folder_name=TEMP_DIR_NAME)

    train_rows = summary_df[:split_idx]
    train_df_file = os.path.join(TEMP_DIR_NAME, f'train_split_{dataset_summary_file}')
    train_rows.to_csv(os.path.join(raw_dir_path, train_df_file), index=False)

    val_rows = summary_df[split_idx:]
    val_df_file = os.path.join(TEMP_DIR_NAME, f'val_split_{dataset_summary_file}')
    val_rows.to_csv(os.path.join(raw_dir_path, val_df_file), index=False)

    return train_df_file, val_df_file

def get_trainer_config(model_name: str, config: dict, logger: Logger = None) -> dict:
    def log(msg):
        if logger:
            logger.log(msg)

    EDGE_MODELS = ['EdgeGNNAttn']
    trainer_params = {}

    train_config = config['training_parameters']
    loss_func_parameters = config['loss_func_parameters']

    # Base Trainer parameters
    node_loss_func = loss_func_parameters['node_loss']
    edge_loss_func = loss_func_parameters['edge_loss']
    node_criterion = get_loss_func(node_loss_func, **loss_func_parameters.get(node_loss_func, {}))
    edge_criterion = get_loss_func(edge_loss_func, **loss_func_parameters.get(edge_loss_func, {}))
    loss_func = edge_criterion if model_name in EDGE_MODELS else node_criterion

    early_stopping_patience = train_config['early_stopping_patience']
    num_epochs = train_config['num_epochs']
    num_epochs_dyn_loss = train_config['num_epochs_dyn_loss']
    log(f'Using dynamic loss weight adjustment for the first {num_epochs_dyn_loss}/{num_epochs} epochs')
    base_config = {
        'num_epochs': num_epochs,
        'num_epochs_dyn_loss': num_epochs_dyn_loss,
        'batch_size': train_config['batch_size'],
        'gradient_clip_value': train_config['gradient_clip_value'],
        'loss_func': loss_func,
        'early_stopping_patience': early_stopping_patience,
    }
    log(f'Using training configuration: {base_config}')
    trainer_params.update(base_config)

    # Physics-informed training parameters
    if model_name not in EDGE_MODELS:
        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        global_mass_loss_scale = loss_func_parameters['global_mass_loss_scale']
        if use_global_mass_loss:
            log(f'Using global mass conservation loss with scale {global_mass_loss_scale}')

        use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
        local_mass_loss_scale = loss_func_parameters['local_mass_loss_scale']
        if use_local_mass_loss:
            log(f'Using local mass conservation loss with scale {local_mass_loss_scale}')

        trainer_params.update({
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_scale': global_mass_loss_scale,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_scale': local_mass_loss_scale,
        })

    # Autoregressive training parameters
    autoregressive_train_config = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_config.get('enabled', False)
    if autoregressive_enabled:
        init_num_timesteps = autoregressive_train_config['init_num_timesteps']
        total_num_timesteps = autoregressive_train_config['total_num_timesteps']
        learning_rate_decay = autoregressive_train_config['learning_rate_decay']
        max_curriculum_epochs = autoregressive_train_config['max_curriculum_epochs']
        log(f'Using autoregressive training for {init_num_timesteps}/{total_num_timesteps} timesteps and curriculum learning with patience {early_stopping_patience}, max {max_curriculum_epochs} epochs and learning rate decay {learning_rate_decay}')

        trainer_params.update({
            'init_num_timesteps': init_num_timesteps,
            'total_num_timesteps': total_num_timesteps,
            'learning_rate_decay': learning_rate_decay,
            'max_curriculum_epochs': max_curriculum_epochs,
        })

    # Node/Edge prediction parameters
    if 'NodeEdgeGNN' in model_name:
        edge_pred_loss_scale = loss_func_parameters['edge_pred_loss_scale']
        log(f'Using edge prediction loss with scale {edge_pred_loss_scale}')
        log(f"Using {edge_criterion.__class__.__name__} loss for edge prediction")
        trainer_params.update({
            'edge_loss_func': edge_criterion,
            'edge_pred_loss_scale': edge_pred_loss_scale,
        })

    return trainer_params

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
