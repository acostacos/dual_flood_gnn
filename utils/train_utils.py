import os
import pandas as pd

from typing import Tuple

def split_dataset_events(root_dir: str, dataset_summary_file: str, percent_validation: float) -> Tuple[str, str]:
    if not (0 < percent_validation < 1):
        raise ValueError(f'Invalid percent_split: {percent_validation}. Must be between 0 and 1.')

    raw_dir_path = os.path.join(root_dir, 'raw')
    dataset_summary_path = os.path.join(raw_dir_path, dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

    split_idx = len(summary_df) - int(len(summary_df) * percent_validation)

    train_rows = summary_df[:split_idx]
    train_df_file = f'train_split_{dataset_summary_file}'
    train_rows.to_csv(os.path.join(raw_dir_path, train_df_file), index=False)

    val_rows = summary_df[split_idx:]
    val_df_file = f'val_split_{dataset_summary_file}'
    val_rows.to_csv(os.path.join(raw_dir_path, val_df_file), index=False)

    return train_df_file, val_df_file
