import os
import numpy as np
import traceback
import torch
import random

from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from datetime import datetime
from itertools import product
from torch.nn import MSELoss
from training import NodeRegressionTrainer, DualRegressionTrainer, DualAutoregressiveTrainer
from testing import DualAutoregressiveTester, NodeAutoregressiveTester
from typing import Dict, Tuple, Optional, List
from utils import Logger, file_utils
from utils.hp_search_utils import HYPERPARAMETER_CHOICES, load_datasets, load_model,\
    create_cross_val_dataset_files, create_temp_dirs, delete_temp_dirs, get_static_config

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--hyperparameters", type=str, choices=HYPERPARAMETER_CHOICES, nargs='+', required=True, help='Hyperparameters to search for')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--summary_file", type=str, required=True, help='Dataset summary file for hyperparameter search. Events in file will be used for cross-validation')
    parser.add_argument("--top_k", type=int, default=5, help='Number of top results to keep for hyperparameter search')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    return parser.parse_args()

def get_hyperparam_search_config() -> Tuple[bool, bool, bool]:
    hyperparams_to_search = args.hyperparameters

    use_global_mass_loss = 'global_mass_loss' in hyperparams_to_search
    use_local_mass_loss = 'local_mass_loss' in hyperparams_to_search
    use_edge_pred_loss = 'edge_pred_loss' in hyperparams_to_search
    return use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss

def get_hyperparameters_for_search(hyperparam_comb: Tuple) -> Dict:
    index = 0

    global_mass_loss_percent = None
    if use_global_mass_loss:
        global_mass_loss_percent = hyperparam_comb[index]
        index += 1

    local_mass_loss_percent = None
    if use_local_mass_loss:
        local_mass_loss_percent = hyperparam_comb[index]
        index += 1

    edge_pred_loss_percent = None
    if use_edge_pred_loss:
        edge_pred_loss_percent = hyperparam_comb[index]
        index += 1
    return global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent

def save_cross_val_results(trainer: DualAutoregressiveTrainer,
                           tester: DualAutoregressiveTester,
                           model_postfix: str):
    curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f'{args.model}_{curr_date_str}{model_postfix}'
    stats_dir = train_config['stats_dir']
    if stats_dir is not None:
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        saved_metrics_path = os.path.join(stats_dir, f'{model_name}_train_stats.npz')
        trainer.save_stats(saved_metrics_path)

    output_dir = test_config['output_dir']
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tester.save_stats(output_dir, stats_filename_prefix=model_name)

def cross_validate(global_mass_loss_percent: Optional[float],
                   local_mass_loss_percent: Optional[float],
                   edge_pred_loss_percent: Optional[float],
                   save_stats_for_first: bool = False) -> float | Tuple[float, float]:
    val_rmses = []
    if use_edge_pred_loss:
        val_edge_rmses = []
    for i, group_id in enumerate(cross_val_groups):
        logger.log(f'Cross-validating with Group {group_id} as the test set...\n')

        storage_mode = config['dataset_parameters']['storage_mode']
        train_dataset, test_dataset = load_datasets(group_id,
                                                    base_dataset_config,
                                                    use_global_mass_loss,
                                                    use_local_mass_loss,
                                                    storage_mode)
        model = load_model(args.model, model_params, train_dataset, args.device)
        delta_t = train_dataset.timestep_interval

        # ============ Training Phase ============
        criterion = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

        loss_func_config = config['loss_func_parameters']
        if global_mass_loss_percent is None:
            global_mass_loss_percent = loss_func_config['global_mass_loss_percent']
        if local_mass_loss_percent is None:
            local_mass_loss_percent = loss_func_config['local_mass_loss_percent']
        if edge_pred_loss_percent is None:
            edge_pred_loss_percent = loss_func_config['edge_pred_loss_percent']
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'optimizer': optimizer,
            'loss_func': criterion,
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_scale': loss_func_config['global_mass_loss_scale'],
            'global_mass_loss_percent': global_mass_loss_percent,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_scale': loss_func_config['local_mass_loss_scale'],
            'local_mass_loss_percent': local_mass_loss_percent,
            'delta_t': delta_t,
            'batch_size': train_config['batch_size'],
            'num_epochs': train_config['num_epochs'],
            'num_epochs_dyn_loss': train_config['num_epochs_dyn_loss'],
            'gradient_clip_value': train_config['gradient_clip_value'],
            'logger': None,
            'device': args.device,
        }
        if use_edge_pred_loss:
            trainer_params.update({
                'edge_pred_loss_scale': loss_func_config['edge_pred_loss_scale'],
                'edge_pred_loss_percent': edge_pred_loss_percent,
            })

            # TODO: Implement autoregressive training for hp search
            autoregressive_train_params = train_config['autoregressive']
            if autoregressive_train_params.get('enabled', False):
                num_timesteps = train_config['autoregressive_timesteps']
                curriculum_epochs = train_config['curriculum_epochs']
                logger.log(f'Using autoregressive training with intervals of {num_timesteps} timessteps and curriculum learning for {curriculum_epochs} epochs')

                trainer = DualAutoregressiveTrainer(**trainer_params, num_timesteps=num_timesteps, curriculum_epochs=curriculum_epochs)
            else:
                trainer = DualRegressionTrainer(**trainer_params)
        else:
            trainer = NodeRegressionTrainer(**trainer_params)

        with open(os.devnull, "w") as f, redirect_stdout(f):
            trainer.train()
        trainer.training_stats.log = logger.log  # Restore logging to console
        trainer.print_stats_summary()

        # ============ Testing Phase ============
        logger.log('\nValidating model...')

        tester_params = {
            'model': model,
            'dataset': test_dataset,
            'rollout_start': test_config['rollout_start'],
            'rollout_timesteps': test_config['rollout_timesteps'],
            'include_physics_loss': False,
            'logger': None,
            'device': args.device,
        }
        if use_edge_pred_loss:
            tester = DualAutoregressiveTester(**tester_params)
        else:
            tester = NodeAutoregressiveTester(**tester_params)

        with open(os.devnull, "w") as f, redirect_stdout(f):
            tester.test()

        avg_rmse = tester.get_avg_node_rmse()
        val_rmses.append(avg_rmse)
        logger.log(f'Group {group_id} RMSE: {avg_rmse:.4e}')

        if use_edge_pred_loss:
            avg_edge_rmse = tester.get_avg_edge_rmse()
            val_edge_rmses.append(avg_edge_rmse)
            logger.log(f'Group {group_id} Edge RMSE: {avg_edge_rmse:.4e}')

        # ============ Saving stats (optional) ============
        if save_stats_for_first and i == 0:
            model_postfix = ''
            if use_global_mass_loss:
                model_postfix += f'_g{global_mass_loss_percent}'
            if use_local_mass_loss:
                model_postfix += f'_l{local_mass_loss_percent}'
            if use_edge_pred_loss:
                model_postfix += f'_e{edge_pred_loss_percent}'

            save_cross_val_results(trainer, tester, model_postfix)

    def get_avg_rmse(rmses: List[float]) -> float:
        np_rmses = np.array(rmses)
        is_not_finite = ~np.isfinite(np_rmses)
        np_rmses[is_not_finite] = 1e10  # Replace non-finite values with a large number
        return np_rmses.mean()

    avg_val_rmse = get_avg_rmse(val_rmses)
    logger.log(f'\nAverage RMSE across all events: {avg_val_rmse:.4e}')
    if not use_edge_pred_loss:
        return avg_val_rmse

    avg_val_edge_rmse = get_avg_rmse(val_edge_rmses)
    logger.log(f'Average Edge RMSE across all events: {avg_val_edge_rmse:.4e}')
    return avg_val_rmse, avg_val_edge_rmse

def search(hyperparameters: Dict[str, List[float]]):
    hp_stats = []
    best_rmses = np.full(args.top_k, np.inf)
    if use_edge_pred_loss:
        best_edge_rmses = np.full(args.top_k, np.inf)
    best_hyperparameters = [{}] * args.top_k

    hyperparameter_list = list(hyperparameters.keys())
    hyperparameter_values = [hyperparameters[key] for key in hyperparameter_list]
    combinations = list(product(*hyperparameter_values))
    for comb in combinations:
        logger.log('\nRunning training with hyperparameter combination:')
        for key, value in zip(hyperparameter_list, comb):
            logger.log(f'\t{key}: {value}')

        search_hyperparams = get_hyperparameters_for_search(comb)
        global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent = search_hyperparams

        results = cross_validate(global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent)
        hp_stats.append({ 'hyperparameters': comb, 'validation_loss': results })

        if use_edge_pred_loss:
            avg_val_rmse, avg_val_edge_rmse = results
            is_best = avg_val_rmse < best_rmses.max() and avg_val_edge_rmse < best_edge_rmses.max()
            worst_idx = np.argmax(best_rmses) if avg_val_rmse < best_rmses.max() else np.argmax(best_edge_rmses)
        else:
            avg_val_rmse = results
            is_best = avg_val_rmse < best_rmses.max()
            worst_idx = np.argmax(best_rmses)

        if is_best:
            best_rmses[worst_idx] = avg_val_rmse
            if use_edge_pred_loss:
                best_edge_rmses[worst_idx] = avg_val_edge_rmse

            new_best_hp = {}
            if use_global_mass_loss:
                new_best_hp['global_mass_loss_percent'] = global_mass_loss_percent
            if use_local_mass_loss:
                new_best_hp['local_mass_loss_percent'] = local_mass_loss_percent
            if use_edge_pred_loss:
                new_best_hp['edge_pred_loss_percent'] = edge_pred_loss_percent
            best_hyperparameters[worst_idx] = new_best_hp

            logger.log(f'New best RMSE for hyperparameter combination:')
            for key, value in zip(hyperparameter_list, comb):
                logger.log(f'\t{key}: {value}')

    output_dir = test_config['output_dir']
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_path = os.path.join(output_dir, f'hp_search_results_{curr_date_str}.npz')
        np.savez(results_path, hp_stats=hp_stats)

    if use_edge_pred_loss:
        return (best_rmses, best_edge_rmses), best_hyperparameters
    return best_rmses, best_hyperparameters

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']

    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Check which hyperparameters will be searched
        use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss = get_hyperparam_search_config()
        assert use_global_mass_loss or use_local_mass_loss or use_edge_pred_loss, 'At least one hyperparameter must be selected for search'
        assert not use_edge_pred_loss or 'NodeEdgeGNN' in args.model, 'Edge prediction loss can only be used with NodeEdgeGNN model'

        hyperparameters = {}
        if use_global_mass_loss:
            GLOBAL_LOSS_PERCENTS = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
            hyperparameters['global_mass_loss_percent'] = GLOBAL_LOSS_PERCENTS
        if use_local_mass_loss:
            LOCAL_LOSS_PERCENTS = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
            hyperparameters['local_mass_loss_percent'] = LOCAL_LOSS_PERCENTS
        if use_edge_pred_loss:
            EDGE_LOSS_PERCENTS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            hyperparameters['edge_pred_loss_percent'] = EDGE_LOSS_PERCENTS

        logger.log(f'Performing hyperparameter search for the following hyperparameters: {', '.join(list(hyperparameters.keys()))}')

        static_config = get_static_config(config, args.model, logger)
        base_dataset_config, model_params, train_config, test_config = static_config

        # Begin hyperparameter search
        root_dir = base_dataset_config['root_dir']
        raw_temp_dir_path, processed_temp_dir_path = create_temp_dirs(root_dir)
        cross_val_groups = create_cross_val_dataset_files(root_dir, args.summary_file)

        best_rmse_values, best_hyperparameters = search(hyperparameters)

        if use_edge_pred_loss:
            best_rmses, best_edge_rmses = best_rmse_values
        else:
            best_rmses = best_rmse_values

        for i in range(len(best_rmses)):
            logger.log(f'Best hyperparameter combination {i+1}:')
            logger.log(f'\tRMSE: {best_rmses[i]:.4e}')
            if use_edge_pred_loss:
                logger.log(f'\tEdge RMSE: {best_edge_rmses[i]:.4e}')
            logger.log('\tHyperparameters:')
            for key, value in best_hyperparameters[i].items():
                logger.log(f'\t\t{key}: {value}')

        # Clean up temporary directories
        delete_temp_dirs(raw_temp_dir_path, processed_temp_dir_path)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')