import os
import numpy as np

from numpy import ndarray
from utils.file_utils import read_yaml_file, save_to_yaml_file
from typing import Dict, List, Literal, Tuple

class DatasetNormalizer:
    EPS = 1e-7 # Prevent division by zero

    def __init__(self, mode: Literal['train', 'test'], root_dir: str, features_stats_file: str):
        self.mode = mode
        self.feature_stats_path = os.path.join(root_dir, 'processed', features_stats_file)
        self.feature_stats = self.load_feature_stats()
 
    def load_feature_stats(self) -> Dict:
        if not os.path.exists(self.feature_stats_path):
            return {}

        feature_stats = read_yaml_file(self.feature_stats_path)
        return feature_stats

    def save_feature_stats(self):
        save_to_yaml_file(self.feature_stats_path, self.feature_stats)

    def get_feature_mean_std(self, feature: str) -> Tuple[float, float]:
        """Get the mean and std of a feature"""
        if feature not in self.feature_stats:
            raise ValueError(f'Feature {feature} not found in feature stats.')

        mean = self.feature_stats[feature]['mean']
        std = self.feature_stats[feature]['std']
        return mean, std

    def update_stats(self, feature: str, feature_data: ndarray):
        assert self.mode == 'train', 'Feature statistics can only be saved in training mode.'

        mean = feature_data.mean().item()
        std = feature_data.std().item()
        self.feature_stats[feature] = {'mean': mean, 'std': std}

    def normalize_feature_vector(self, feature_list: List[str], feature_vector: ndarray) -> ndarray:
        normalized_vector = []
        is_dynamic_feature = len(feature_vector.shape) == 3
        for i, feature in enumerate(feature_list):
            feature_data = feature_vector[:, :, i:i+1] if is_dynamic_feature else feature_vector[:, i:i+1]

            if feature not in self.feature_stats:
                raise ValueError(f'Feature {feature} not found in feature stats.')

            mean = self.feature_stats[feature]['mean']
            std = self.feature_stats[feature]['std']
            feature_data = self.normalize(feature_data, mean, std)
            normalized_vector.append(feature_data)

        return np.concat(normalized_vector, axis=-1)

    def get_normalized_zero_tensor(self, feature_list: List[str], other_dims: Tuple[int, ...], dtype: np.dtype = np.float32) -> ndarray:
        out_tensor = []
        shape = (*other_dims, 1)
        for feature in feature_list:
            zeros = np.zeros(shape, dtype=dtype)

            if feature not in self.feature_stats:
                raise ValueError(f'Feature {feature} not found in feature stats when creating normalized zeros array.')

            mean = self.feature_stats[feature]['mean']
            std = self.feature_stats[feature]['std']
            zeros = self.normalize(zeros, mean, std)

            out_tensor.append(zeros)
        return np.concat(out_tensor, axis=-1)

    def normalize(self, feature_data: ndarray, mean: float, std: float) -> ndarray:
        """Z-score normalization of features"""
        return (feature_data - mean) / (std + DatasetNormalizer.EPS)

    def denormalize(self, feature: str, feature_data: ndarray) -> ndarray:
        """Z-score denormalization of features"""
        if feature not in self.feature_stats:
            raise ValueError(f'Feature {feature} not found in feature stats.')

        mean = self.feature_stats[feature]['mean']
        std = self.feature_stats[feature]['std']
        return feature_data * (std + DatasetNormalizer.EPS) + mean 
