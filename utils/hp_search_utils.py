import os
import pandas as pd
import shutil

from data import FloodEventDataset, InMemoryFloodEventDataset
from models import model_factory
from models.base_model import BaseModel
from typing import Dict, List, Tuple
from utils import Logger

TEMP_DIR_NAME = '_temp_hp_dir'
HYPERPARAMETER_CHOICES = [
    'global_mass_loss',
    'local_mass_loss',
    'edge_pred_loss',
]
dataset_cache = {}

def get_static_config(config: Dict, model_name: str, logger: Logger):
    # ============ Dataset Configuration ============
    dataset_parameters = config['dataset_parameters']
    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'nodes_shp_file': dataset_parameters['nodes_shp_file'],
        'edges_shp_file': dataset_parameters['edges_shp_file'],
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_timesteps': dataset_parameters['spin_up_timesteps'],
        'timesteps_from_peak': dataset_parameters['timesteps_from_peak'],
        'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
        'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        'logger': logger,
        'force_reload': False,
    }
    logger.log(f'Using train dataset configuration: {base_dataset_config}')

    # ============ Model Configuration ============
    model_params = config['model_parameters'][model_name]
    logger.log(f'Using model: {model_name}')
    logger.log(f'Using model configuration: {model_params}')

    # ============ Training Configuration ============
    train_config = config['training_parameters']
    logger.log(f'Using training configuration: {train_config}')

    # ============ Testing Configuration ============
    test_config = config['testing_parameters']
    logger.log(f'Using testing configuration: {test_config}')

    return base_dataset_config, model_params, train_config, test_config

def delete_temp_dirs(raw_temp_dir_path: str, processed_temp_dir_path: str):
    if os.path.exists(raw_temp_dir_path):
        shutil.rmtree(raw_temp_dir_path)
    if os.path.exists(processed_temp_dir_path):
        shutil.rmtree(processed_temp_dir_path)

def create_temp_dirs(root_dir: str):
    raw_temp_dir_path = os.path.join(root_dir, 'raw', TEMP_DIR_NAME)
    processed_temp_dir_path = os.path.join(root_dir, 'processed', TEMP_DIR_NAME)

    delete_temp_dirs(raw_temp_dir_path, processed_temp_dir_path)

    os.makedirs(processed_temp_dir_path)
    os.makedirs(raw_temp_dir_path)

    return raw_temp_dir_path, processed_temp_dir_path

def create_cross_val_dataset_files(root_dir: str, dataset_summary_file: str) -> List[str]:
    dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'
    assert 'Group' in summary_df.columns, f'Missing Group column in summary file: {dataset_summary_path}'

    groups = summary_df['Group'].unique()

    raw_temp_dir_path = os.path.join(root_dir, 'raw', TEMP_DIR_NAME)
    for group in groups:
        other_rows = summary_df[summary_df['Group'] != group]
        train_df_path = os.path.join(raw_temp_dir_path, f'train_{group}.csv')
        other_rows.to_csv(train_df_path, index=False)

        group_rows = summary_df[summary_df['Group'] == group]
        test_df_path = os.path.join(raw_temp_dir_path, f'test_{group}.csv')
        group_rows.to_csv(test_df_path, index=False)

    return groups

def load_datasets(
        run_id: str,
        base_dataset_config: Dict,
        use_global_mass_loss: bool,
        use_local_mass_loss: bool,
        storage_mode: str) -> Tuple[FloodEventDataset, FloodEventDataset]:
    if f'train_{run_id}' in dataset_cache and f'test_{run_id}' in dataset_cache:
        return dataset_cache[f'train_{run_id}'], dataset_cache[f'test_{run_id}']

    features_stats_file = os.path.join(TEMP_DIR_NAME, f'features_stats_{run_id}.yaml')
    train_dataset_summary_file = os.path.join(TEMP_DIR_NAME, f'train_{run_id}.csv')
    train_event_stats_file = os.path.join(TEMP_DIR_NAME, f'train_event_stats_{run_id}.yaml')
    train_dataset_config = {
        **base_dataset_config,
        'mode': 'train',
        'dataset_summary_file': train_dataset_summary_file,
        'event_stats_file': train_event_stats_file,
        'features_stats_file': features_stats_file,
        'with_global_mass_loss': use_global_mass_loss,
        'with_local_mass_loss': use_local_mass_loss,
    }

    test_dataset_summary_file = os.path.join(TEMP_DIR_NAME, f'test_{run_id}.csv')
    test_event_stats_file = os.path.join(TEMP_DIR_NAME, f'test_event_stats_{run_id}.yaml')
    test_dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': test_dataset_summary_file,
        'event_stats_file': test_event_stats_file,
        'features_stats_file': features_stats_file,
        # Exclude computation of physics loss for hyperparameter search
        'with_global_mass_loss': False,
        'with_local_mass_loss': False,
    }

    dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset

    train_dataset = dataset_class(**train_dataset_config)
    dataset_cache[f'train_{run_id}'] = train_dataset

    test_dataset = dataset_class(**test_dataset_config)
    dataset_cache[f'test_{run_id}'] = test_dataset

    return train_dataset, test_dataset

def load_model(model_name: str, model_params: Dict, dataset: FloodEventDataset, device: str) -> BaseModel:
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
