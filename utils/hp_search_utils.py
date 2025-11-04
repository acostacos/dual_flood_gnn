import os
import pandas as pd
import optuna

from data import dataset_factory, FloodEventDataset
from models import model_factory
from models.base_model import BaseModel
from typing import Dict, List, Tuple, Optional
from utils import Logger, file_utils

TEMP_DIR_NAME = 'hp_search_cross_val'
dataset_cache = {}

def create_cross_val_dataset_files(root_dir: str,
                                   dataset_summary_file: str,
                                   num_folds: int,
                                   percent_validation: Optional[float] = None) -> Tuple[List[str], List[str]]:
    raw_dir_path = os.path.join(root_dir, 'raw')
    processed_dir_path = os.path.join(root_dir, 'processed')
    temp_dir_paths = file_utils.create_temp_dirs(paths=[raw_dir_path, processed_dir_path],
                                                 folder_name=TEMP_DIR_NAME)

    dataset_summary_path = os.path.join(raw_dir_path, dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'
    assert len(summary_df) >= num_folds, f'Number of flood events ({len(summary_df)}) must be greater than or equal to number of folds ({num_folds})'

    num_events_per_fold = len(summary_df) // num_folds
    groups = []
    for fold in range(num_folds):
        group_id = f'fold{fold+1}'
        groups.append(group_id)
        start_idx = fold * num_events_per_fold
        if fold == num_folds - 1:
            end_idx = len(summary_df)
        else:
            end_idx = (fold + 1) * num_events_per_fold
        summary_df.loc[start_idx:end_idx, 'Group'] = group_id

    raw_temp_dir_path = temp_dir_paths[0]
    for group in groups:
        other_rows = summary_df[summary_df['Group'] != group]

        if percent_validation is not None:
            num_val_events = max(1, int(len(other_rows) * percent_validation))
            val_rows = other_rows[-num_val_events:]
            other_rows = other_rows.drop(val_rows.index)
            val_df_path = os.path.join(raw_temp_dir_path, f'val_{group}.csv')
            val_rows.to_csv(val_df_path, index=False)

        train_df_path = os.path.join(raw_temp_dir_path, f'train_{group}.csv')
        other_rows.to_csv(train_df_path, index=False)

        group_rows = summary_df[summary_df['Group'] == group]
        test_df_path = os.path.join(raw_temp_dir_path, f'test_{group}.csv')
        group_rows.to_csv(test_df_path, index=False)

    return groups, temp_dir_paths

def suggest_hyperparamters(trial: optuna.Trial, hyperparameters: Dict, config: Dict, logger: Logger = None) -> Dict:
    updated_config = config.copy()
    for param_name, param_info in hyperparameters.items():
        param_type = param_info['type']
        if param_type == 'int':
            suggested_value = trial.suggest_int(param_name, param_info['min'], param_info['max'], step=param_info.get('step', 1), log=param_info.get('log', False))
        elif param_type == 'float':
            suggested_value = trial.suggest_float(param_name, param_info['min'], param_info['max'], step=param_info.get('step', None), log=param_info.get('log', False))
        elif param_type == 'categorical':
            suggested_value = trial.suggest_categorical(param_name, param_info['choices'])
        else:
            raise ValueError(f'Unsupported hyperparameter type: {param_type} for parameter: {param_name}')

        if logger is not None:
            logger.log(f'Testing value {suggested_value} for parameter {param_name}')

        # Traverse the config dictionary to set the suggested value
        path = param_info['path']
        keys = path.split('.')
        d = updated_config
        for key in keys[:-1]:
            if key not in d:
                raise KeyError(f'Key {key} not found in configuration path: {path}')
            d = d[key]
        d[keys[-1]] = suggested_value

    return updated_config

def load_datasets(group_id: str, config: Dict, logger: Logger) -> Tuple[FloodEventDataset, FloodEventDataset, Optional[FloodEventDataset]]:
    train_config = config['training_parameters']
    early_stopping_patience = train_config['early_stopping_patience']

    if f'train_{group_id}' in dataset_cache and f'test_{group_id}' in dataset_cache:
        train_dataset = dataset_cache[f'train_{group_id}']
        test_dataset = dataset_cache[f'test_{group_id}']
        if early_stopping_patience is not None:
            if f'val_{group_id}' not in dataset_cache:
                raise ValueError(f'Validation dataset for group {group_id} not found in cache.')
            val_dataaset = dataset_cache[f'val_{group_id}']
            return train_dataset, test_dataset, val_dataaset
        return train_dataset, test_dataset, None

    dataset_parameters = config['dataset_parameters']
    loss_func_parameters = config['loss_func_parameters']
    features_stats_file = os.path.join(TEMP_DIR_NAME, f'features_stats_{group_id}.yaml')
    with_global_mass_loss = loss_func_parameters['use_global_mass_loss']
    with_local_mass_loss = loss_func_parameters['use_local_mass_loss']
    storage_mode = dataset_parameters['storage_mode']
    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'nodes_shp_file': dataset_parameters['nodes_shp_file'],
        'edges_shp_file': dataset_parameters['edges_shp_file'],
        'dem_file': dataset_parameters['dem_file'],
        'features_stats_file': features_stats_file,
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_time': dataset_parameters['spin_up_time'],
        'time_from_peak': dataset_parameters['time_from_peak'],
        'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
        'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        'logger': logger,
        'force_reload': True,
    }

    train_summary_file = os.path.join(TEMP_DIR_NAME, f'train_{group_id}.csv')
    train_event_stats_file = os.path.join(TEMP_DIR_NAME, f'train_event_stats_{group_id}.yaml')
    train_dataset_config = {
        **base_dataset_config,
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        'with_global_mass_loss': with_global_mass_loss,
        'with_local_mass_loss': with_local_mass_loss,
    }
    autoregressive_train_params = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    if autoregressive_enabled:
        train_dataset_config.update({
            'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
        })
    logger.log(f'Using train dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, **train_dataset_config)
    dataset_cache[f'train_{group_id}'] = train_dataset
    logger.log(f'Loaded train dataset with {len(train_dataset)} samples')

    test_summary_file = os.path.join(TEMP_DIR_NAME, f'test_{group_id}.csv')
    test_event_stats_file = os.path.join(TEMP_DIR_NAME, f'test_event_stats_{group_id}.yaml')
    test_dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': test_summary_file,
        'event_stats_file': test_event_stats_file,
        # Exclude computation of physics loss for hyperparameter search
        'with_global_mass_loss': False,
        'with_local_mass_loss': False,
    }
    logger.log(f'Using test dataset configuration: {test_dataset_config}')
    test_dataset = dataset_factory(storage_mode, autoregressive=False, **test_dataset_config)
    dataset_cache[f'test_{group_id}'] = test_dataset
    logger.log(f'Loaded test dataset with {len(test_dataset)} samples')

    if early_stopping_patience is None:
        return train_dataset, test_dataset, None

    val_summary_file = os.path.join(TEMP_DIR_NAME, f'val_{group_id}.csv')
    val_event_stats_file = os.path.join(TEMP_DIR_NAME, f'val_event_stats_{group_id}.yaml')
    val_dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': val_summary_file,
        'event_stats_file': val_event_stats_file,
        # Exclude computation of physics loss for hyperparameter search
        'with_global_mass_loss': False,
        'with_local_mass_loss': False,
    }
    logger.log(f'Using validation dataset configuration: {val_dataset_config}')
    val_dataset = dataset_factory(storage_mode, autoregressive=False, **val_dataset_config)
    dataset_cache[f'val_{group_id}'] = val_dataset
    logger.log(f'Loaded validation dataset with {len(val_dataset)} samples')

    return train_dataset, test_dataset, val_dataset

def load_model(model_name: str, config: Dict, dataset: FloodEventDataset, device: str) -> BaseModel:
    model_params = config['model_parameters'][model_name]
    base_model_params = {
        'static_node_features': dataset.num_static_node_features,
        'dynamic_node_features': dataset.num_dynamic_node_features,
        'static_edge_features': dataset.num_static_edge_features,
        'dynamic_edge_features': dataset.num_dynamic_edge_features,
        'previous_timesteps': dataset.previous_timesteps,
        'device': device,
    }
    model_config = {**model_params, **base_model_params}
    model = model_factory(model_name, **model_config)
    return model
