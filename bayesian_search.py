import os
import numpy as np
import traceback
import torch
import optuna
import random

from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from datetime import datetime
from optuna.visualization import plot_optimization_history, plot_slice, plot_pareto_front
from torch.nn import MSELoss
from training import NodeRegressionTrainer, DualRegressionTrainer, DualAutoregressiveTrainer
from testing import DualAutoregressiveTester, NodeAutoregressiveTester
from typing import List, Optional, Tuple
from utils import Logger, file_utils
from utils.hp_search_utils import HYPERPARAMETER_CHOICES, load_datasets, load_model,\
    create_cross_val_dataset_files, create_temp_dirs, delete_temp_dirs, get_static_config

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--hyperparameters", type=str, choices=HYPERPARAMETER_CHOICES, nargs='+', required=True, help='Hyperparameters to search for')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--summary_file", type=str, required=True, help='Dataset summary file for hyperparameter search. Events in file will be used for cross-validation')
    parser.add_argument("--num_trials", type=int, default=20, help='Number of trials for hyperparameter search')
    parser.add_argument("--num_folds", type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    return parser.parse_args()

def get_hyperparam_search_config() -> Tuple[bool, bool, bool]:
    hyperparams_to_search = args.hyperparameters

    use_global_mass_loss = 'global_mass_loss' in hyperparams_to_search
    use_local_mass_loss = 'local_mass_loss' in hyperparams_to_search
    use_edge_pred_loss = 'edge_pred_loss' in hyperparams_to_search
    return use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss

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
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['adam_weight_decay'])

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

def objective(trial: optuna.Trial) -> float:
    global_mass_loss_percent = trial.suggest_float('global_mass_loss_percent', 1.0e-6, 0.05, log=True) if use_global_mass_loss else None
    local_mass_loss_percent = trial.suggest_float('local_mass_loss_percent', 1.0e-6, 0.05, log=True) if use_local_mass_loss else None
    edge_pred_loss_percent = trial.suggest_float('edge_pred_loss_percent', 0, 1, step=0.01) if use_edge_pred_loss else None

    logger.log(f'Hyperparameters: global_mass_loss_percent={global_mass_loss_percent}, local_mass_loss_percent={local_mass_loss_percent}, edge_pred_loss_percent={edge_pred_loss_percent}')

    return cross_validate(global_mass_loss_percent, local_mass_loss_percent, edge_pred_loss_percent)

def plot_hyperparameter_search_results(study: optuna.Study):
    stats_dir = train_config['stats_dir']
    if stats_dir is None:
        return

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    if use_edge_pred_loss:
        fig = plot_optimization_history(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_optimization_history.html'))

        fig = plot_optimization_history(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_edge_optimization_history.html'))

        fig = plot_slice(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_slice_plot.html'))

        fig = plot_slice(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_edge_slice_plot.html'))

        fig = plot_pareto_front(study=study, target_names=['Node RMSE', 'Edge RMSE'])
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_pareto_front.html'))
    else:
        fig = plot_optimization_history(study, target_name='RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_optimization_history.html'))

        fig = plot_slice(study, target_name='RMSE')
        fig.write_html(os.path.join(stats_dir, f'{study.study_name}_slice_plot.html'))

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    # Initialize logger
    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss = get_hyperparam_search_config()
        assert use_global_mass_loss or use_local_mass_loss or use_edge_pred_loss, 'At least one hyperparameter must be selected for search'
        assert not use_edge_pred_loss or 'NodeEdgeGNN' in args.model, 'Edge prediction loss can only be used with NodeEdgeGNN model'

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        static_config = get_static_config(config, args.model, logger)
        base_dataset_config, model_params, train_config, test_config = static_config

        # Begin hyperparameter search
        root_dir = base_dataset_config['root_dir']
        raw_temp_dir_path, processed_temp_dir_path = create_temp_dirs(root_dir)
        logger.log(f'Creating {args.num_folds}-fold cross-validation dataset files from {args.summary_file}...')
        cross_val_groups = create_cross_val_dataset_files(root_dir, args.summary_file, args.num_folds)

        study_name = f'{"_".join(args.hyperparameters)}'
        study_kwargs = {'study_name': study_name }
        if use_edge_pred_loss:
            study_kwargs['directions'] = ['minimize', 'minimize']
        else:
            study_kwargs['direction'] = 'minimize'

        study = optuna.create_study(**study_kwargs)
        logger.log(f'Using sampler: {study.sampler.__class__.__name__ if study.sampler else None}')
        logger.log(f'Using pruner: {study.pruner.__class__.__name__ if study.pruner else None}')
        logger.log(f'Running hyperparameter search for {args.num_trials} trials...')
        study.optimize(objective, n_trials=args.num_trials)

        if use_edge_pred_loss:
            logger.log('Best hyperparameters found:')
            for trial in study.best_trials:
                logger.log(f'Trial {trial.number}:')
                for key, value in trial.params.items():
                    logger.log(f'\t{key}: {value}')
                objective_values_str = ', '.join([f'{v:.4e}' for v in trial.values])
                logger.log(f'\tObjective values: {objective_values_str}')
        else:
            logger.log('Best hyperparameters found:')
            for key, value in study.best_params.items():
                logger.log(f'{key}: {value}')
            logger.log(f'Best objective value: {study.best_value:.4e}')

        # Plot hyperparameter search results
        plot_hyperparameter_search_results(study)

        # Clean up temporary directories
        delete_temp_dirs(raw_temp_dir_path, processed_temp_dir_path)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')
