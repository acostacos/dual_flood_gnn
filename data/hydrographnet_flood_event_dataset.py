import os
import torch
import json
import numpy as np

from scipy.spatial import KDTree
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from typing import Optional, Tuple, Union, Literal
from utils import Logger

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"

# Trimmed features similar to our current dataset
STATIC_NODE_FEATURES = ['area', 'elevation', 'slope', 'manning', 'inflow', 'precipitation']
DYNAMIC_NODE_FEATURES = ['volume']
STATIC_EDGE_FEATURES = ['distance']

# Complete features (Need to uncomment in create_edge_features and create_node_features methods)
# STATIC_NODE_FEATURES = ['x_coords', 'y_coords', 'area', 'elevation', 'slope', 'aspect', 'curvature', 'manning', 'flow_accum', 'infiltration', 'inflow', 'precipitation']
# DYNAMIC_NODE_FEATURES = ['water_depth', 'volume']
# STATIC_EDGE_FEATURES = ['relative_coords_x', 'relative_coords_y', 'distance']

def load_hgn_dataset(split: Literal["train", "test"],
                     hydrograph_ids_file: str,
                     num_label_timesteps: Optional[int] = None,
                     logger: Logger = None):
    return HydroGraphNetFloodEventDataset(data_dir="C:/Users/Carlo/Documents/School/Masters/NUS/Dissertation/HydroGraphNet/outputs_phy/data",
                                          prefix="M80",
                                          n_time_steps=2,
                                          k=4,
                                          hydrograph_ids_file=hydrograph_ids_file,
                                          split=split,
                                          num_label_timesteps=num_label_timesteps,
                                          logger=logger)

class HydroGraphNetFloodEventDataset(Dataset):
    '''Adopted from https://github.com/MehdiTaghizadehUVa/modulus/tree/main/examples/weather/flood_modeling/hydrographnet'''
    NODE_TARGET_FEATURE = 'volume'
    EDGE_TARGET_FEATURE = None
    DYNAMIC_NODE_FEATURES = DYNAMIC_NODE_FEATURES
    STATIC_NODE_FEATURES = STATIC_NODE_FEATURES
    DYNAMIC_EDGE_FEATURES = [None]

    def __init__(self,
                 data_dir: str,
                 prefix: str = 'M80',
                 n_time_steps: int = 2,
                 k: int = 4,
                 hydrograph_ids_file: Optional[str] = None,
                 split: Literal["train", "test"] = "train",
                 return_physics: bool = False,
                 rollout_length: Optional[int] = None,
                 num_label_timesteps: Optional[int] = None,
                 logger: Logger = None,
                 debug: bool = False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        # Setup debug and logger
        log = print
        if logger is not None and hasattr(logger, 'log'):
            log = logger.log
        self.log = log

        self.data_dir = data_dir
        self.prefix = prefix
        self.n_time_steps = n_time_steps
        self.k = k
        self.hydrograph_ids_file = hydrograph_ids_file
        self.split = split
        self.return_physics = return_physics

        # rollout_length is only used when split=="test"
        self.rollout_length = rollout_length

        # num_label_timesteps is only used for autoregressive training
        self.num_label_timesteps = num_label_timesteps

        # Initialize dataset variables
        self.static_data = {}
        self.dynamic_data = []
        self.sample_index = []
        self.hydrograph_ids = []
        self.static_stats = {}
        self.dynamic_stats = {}

        self.num_static_node_features = len(STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = 0
        self.previous_timesteps = n_time_steps - 1
        self.event_start_idx = []
        self.total_rollout_timesteps = 0

        self.timestep_interval = 1200
        self.is_normalized = True
        self.boundary_condition = type('AnonymousObject', (object,), {
            'boundary_nodes_mask': None,
            'boundary_edges_mask': None,
            'inflow_edges_mask': None,
        })
        self.normalizer = type('AnonymousObject', (object,), {
            'normalize': lambda feature, data: self.normalize(data,
                                                              self.dynamic_stats[feature]['mean'] if feature in DYNAMIC_NODE_FEATURES else self.static_stats[feature]['mean'],
                                                              self.dynamic_stats[feature]['std'] if feature in DYNAMIC_NODE_FEATURES else self.static_stats[feature]['std']),
            'denormalize': lambda feature, data: self.denormalize(data,
                                                                self.dynamic_stats[feature]['mean'] if feature in DYNAMIC_NODE_FEATURES else self.static_stats[feature]['mean'],
                                                                self.dynamic_stats[feature]['std'] if feature in DYNAMIC_NODE_FEATURES else self.static_stats[feature]['std']),
        })

        super().__init__(data_dir, transform, pre_transform, pre_filter, log=debug)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Data must be downloaded manually and placed in the raw_dir
        pass

    def process(self):
        # Maybe load all static features that are loaded
        if self.split == "train":
            # For training, load constant data and compute static normalization stats.
            (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
             manning, flow_accum, infiltration, self.static_stats) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=None)
            self.save_norm_stats(self.static_stats, STATIC_NORM_STATS_FILE)
        else:
            # For test or validation, load precomputed normalization stats.
            self.static_stats = self.load_norm_stats(STATIC_NORM_STATS_FILE)
            (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
             manning, flow_accum, infiltration, _) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=self.static_stats)

        # Build the graph connectivity using a k-d tree.
        num_nodes = xy_coords.shape[0]
        kdtree = KDTree(xy_coords)
        _, neighbors = kdtree.query(xy_coords, k=self.k + 1)
        edge_index = np.vstack([(i, nbr) for i, nbrs in enumerate(neighbors)
                                  for nbr in nbrs if nbr != i]).T
        edge_features = self.create_edge_features(xy_coords, edge_index)

        # Store static data.
        self.static_data = {
            "xy_coords": xy_coords,
            "area": area,
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "manning": manning,
            "flow_accum": flow_accum,
            "infiltration": infiltration,
            "area_denorm": area_denorm,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

        # Read hydrograph IDs from the file.
        file_path = os.path.join(self.data_dir, self.hydrograph_ids_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hydrograph IDs file not found: {file_path}")

        with open(file_path, "r") as f:
            lines = f.readlines()
        self.hydrograph_ids = [line.strip() for line in lines if line.strip()]
        self.hec_ras_run_ids = self.hydrograph_ids

        # Process dynamic data (water depth, inflow, volume, precipitation) for each hydrograph.
        temp_dynamic_data = []
        water_depth_list = []
        volume_list = []
        precipitation_list = []
        inflow_list = []
        for hid in tqdm(self.hydrograph_ids, desc="Processing Hydrographs"):
            water_depth, inflow_hydrograph, volume, precipitation = self.load_dynamic_data(
                self.data_dir, hid, self.prefix, num_points=num_nodes)
            temp_dynamic_data.append({
                "water_depth": water_depth,
                "inflow_hydrograph": inflow_hydrograph,
                "volume": volume,
                "precipitation": precipitation,
                "hydro_id": hid,
            })
            water_depth_list.append(water_depth.flatten())
            volume_list.append(volume.flatten())
            precipitation_list.append(precipitation.flatten())
            inflow_list.append(inflow_hydrograph.flatten())

        # Compute dynamic normalization statistics for training or load precomputed stats.
        if self.split == "train":
            self.dynamic_stats = {}
            water_depth_all = np.concatenate(water_depth_list)
            self.dynamic_stats["water_depth"] = {"mean": float(np.mean(water_depth_all)),
                                                 "std": float(np.std(water_depth_all))}
            volume_all = np.concatenate(volume_list)
            self.dynamic_stats["volume"] = {"mean": float(np.mean(volume_all)),
                                            "std": float(np.std(volume_all))}
            precipitation_all = np.concatenate(precipitation_list)
            self.dynamic_stats["precipitation"] = {"mean": float(np.mean(precipitation_all)),
                                                   "std": float(np.std(precipitation_all))}
            inflow_all = np.concatenate(inflow_list)
            self.dynamic_stats["inflow_hydrograph"] = {"mean": float(np.mean(inflow_all)),
                                                       "std": float(np.std(inflow_all))}
            self.save_norm_stats(self.dynamic_stats, DYNAMIC_NORM_STATS_FILE)
        else:
            self.dynamic_stats = self.load_norm_stats(DYNAMIC_NORM_STATS_FILE)

        # Normalize the dynamic data.
        self.dynamic_data = []
        for dyn in temp_dynamic_data:
            dyn_std = {
                "water_depth": self.normalize(dyn["water_depth"],
                                              self.dynamic_stats["water_depth"]["mean"],
                                              self.dynamic_stats["water_depth"]["std"]),
                "volume": self.normalize(dyn["volume"],
                                         self.dynamic_stats["volume"]["mean"],
                                         self.dynamic_stats["volume"]["std"]),
                "precipitation": self.normalize(dyn["precipitation"],
                                                self.dynamic_stats["precipitation"]["mean"],
                                                self.dynamic_stats["precipitation"]["std"]),
                "inflow_hydrograph": self.normalize(dyn["inflow_hydrograph"],
                                                    self.dynamic_stats["inflow_hydrograph"]["mean"],
                                                    self.dynamic_stats["inflow_hydrograph"]["std"]),
                "hydro_id": dyn["hydro_id"],
            }
            self.dynamic_data.append(dyn_std)

        # Build sample indices for training (sliding window) or validate test data.
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                self.event_start_idx.append(len(self.sample_index))

                T = dyn["water_depth"].shape[0]
                max_t = T - self.n_time_steps
                if self.num_label_timesteps is not None:
                    max_t -= self.num_label_timesteps
                for t in range(max_t):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
        elif self.split == "test":
            for h_idx, dyn in enumerate(self.dynamic_data):
                self.event_start_idx.append(len(self.sample_index))

                T = dyn["water_depth"].shape[0]
                rollout = self.rollout_length if self.rollout_length is not None else (T - self.n_time_steps)
                if T < self.n_time_steps + rollout:
                    raise ValueError(
                        f"Hydrograph {dyn['hydro_id']} does not have enough time steps for the specified rollout_length."
                    )
                for t in range(rollout):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
        self.total_rollout_timesteps = self.length
        
        # Add boundary masks (all False for now, can be updated later)
        self.boundary_condition.boundary_nodes_mask = np.zeros((num_nodes,), dtype=bool)
        self.boundary_condition.boundary_edges_mask = np.zeros((edge_index.shape[1],), dtype=bool)
        self.boundary_condition.inflow_edges_mask = np.zeros((edge_index.shape[1],), dtype=bool)

    def len(self) -> int:
        return self.length

    def get(self, idx: int) -> Data:
        sd = self.static_data
        hydro_idx, t_idx = self.sample_index[idx]
        dyn = self.dynamic_data[hydro_idx]
        is_autoregressive = self.split == "train" and self.num_label_timesteps is not None

        # Determine the end index for the dynamic window.
        end_index = t_idx + self.n_time_steps

        # Compute node features and future flow/precipitation values.
        if not is_autoregressive:
            node_features, future_flow, future_precip = self.create_node_features(
                sd["xy_coords"], sd["area"], sd["elevation"], sd["slope"], sd["aspect"],
                sd["curvature"], sd["manning"], sd["flow_accum"], sd["infiltration"],
                dyn["water_depth"][t_idx:end_index, :],
                dyn["volume"][t_idx:end_index, :],
                dyn["precipitation"],
                t_idx, self.n_time_steps, dyn["inflow_hydrograph"]
            )
        else:
            node_features = []
            for step in range(self.num_label_timesteps):
                step_features, _, _ = self.create_node_features(
                    sd["xy_coords"], sd["area"], sd["elevation"], sd["slope"], sd["aspect"],
                    sd["curvature"], sd["manning"], sd["flow_accum"], sd["infiltration"],
                    dyn["water_depth"][t_idx + step:end_index + step, :],
                    dyn["volume"][t_idx + step:end_index + step, :],
                    dyn["precipitation"],
                    t_idx + step, self.n_time_steps, dyn["inflow_hydrograph"]
                )
                node_features.append(step_features)
            node_features = np.stack(node_features, axis=-1)

        # Compute target
        target_time = t_idx + self.n_time_steps
        prev_time = target_time - 1
        # Compute target differences for water depth and volume.
        # target_depth = dyn["water_depth"][target_time, :] - dyn["water_depth"][prev_time, :]
        # target_volume = dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
        # target = np.stack([target_depth, target_volume], axis=1)
        if not is_autoregressive:
            target = dyn["volume"][target_time, :][:, None]
        else:
            target = dyn["volume"][target_time:target_time + self.num_label_timesteps, :].T[:, None, :]

        # Compute edge features
        edge_attr = torch.Tensor(sd["edge_features"]).to(torch.float32)
        if is_autoregressive:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, self.num_label_timesteps)

        # Create the graph with PyTorch Geometric.
        x = torch.Tensor(node_features).to(torch.float32)
        edge_index = torch.Tensor(sd["edge_index"]).to(torch.int64)
        y = torch.Tensor(target).to(torch.float32)
        g = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y, timestep=None)

        if self.split == 'test':
            return g

        if not self.return_physics:
            return g

        # Return physics data if training and return_physics is True.
        # Compute physics data in the denormalized domain.
        past_volume = float(np.sum(dyn["volume"][prev_time, :]))
        future_volume = (float(np.sum(dyn["volume"][target_time + 1, :]))
                            if (target_time + 1 < dyn["volume"].shape[0])
                            else float(np.sum(dyn["volume"][target_time, :])))
        avg_inflow_norm = float((dyn["inflow_hydrograph"][prev_time] + dyn["inflow_hydrograph"][target_time]) / 2)
        avg_precip_norm = float((dyn["precipitation"][prev_time] + dyn["precipitation"][target_time]) / 2)
        denorm_avg_inflow = (avg_inflow_norm *
                                self.dynamic_stats["inflow_hydrograph"]["std"] +
                                self.dynamic_stats["inflow_hydrograph"]["mean"])
        denorm_avg_precip = (avg_precip_norm *
                                self.dynamic_stats["precipitation"]["std"] +
                                self.dynamic_stats["precipitation"]["mean"])

        # --- New: Compute next-step inflow and precipitation for physics loss term2 ---
        if (target_time + 1) < dyn["inflow_hydrograph"].shape[0]:
            next_inflow_norm = dyn["inflow_hydrograph"][target_time + 1]
            next_precip_norm = dyn["precipitation"][target_time + 1]
        else:
            next_inflow_norm = dyn["inflow_hydrograph"][target_time]
            next_precip_norm = dyn["precipitation"][target_time]
        denorm_next_inflow = (next_inflow_norm *
                                self.dynamic_stats["inflow_hydrograph"]["std"] +
                                self.dynamic_stats["inflow_hydrograph"]["mean"])
        denorm_next_precip = (next_precip_norm *
                                self.dynamic_stats["precipitation"]["std"] +
                                self.dynamic_stats["precipitation"]["mean"])

        # Build the complete physics data dictionary.
        full_physics_data = {
            "flow_future": float(future_flow * self.dynamic_stats["inflow_hydrograph"]["std"] +
                                    self.dynamic_stats["inflow_hydrograph"]["mean"]),
            "precip_future": float(future_precip * self.dynamic_stats["precipitation"]["std"] +
                                        self.dynamic_stats["precipitation"]["mean"]),
            "past_volume": past_volume,
            "future_volume": future_volume,
            "avg_inflow": denorm_avg_inflow,
            "avg_precipitation": denorm_avg_precip,
            "next_inflow": denorm_next_inflow,
            "next_precip": denorm_next_precip,
            "volume_mean": float(self.dynamic_stats["volume"]["mean"]),
            "volume_std": float(self.dynamic_stats["volume"]["std"]),
            "inflow_mean": float(self.dynamic_stats["inflow_hydrograph"]["mean"]),
            "inflow_std": float(self.dynamic_stats["inflow_hydrograph"]["std"]),
            "precip_mean": float(self.dynamic_stats["precipitation"]["mean"]),
            "precip_std": float(self.dynamic_stats["precipitation"]["std"]),
            "num_nodes": float(sd["xy_coords"].shape[0]),
            "area_sum": float(np.sum(sd["area_denorm"])),
            "infiltration_area_sum": float(np.sum(
                self.denormalize(sd["infiltration"],
                                    self.static_stats["infiltration"]["mean"],
                                    self.static_stats["infiltration"]["std"]) * sd["area_denorm"]
            )) / 100.0
        }
        return g, full_physics_data

    def save_norm_stats(self, stats: dict, filename: str) -> None:
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(stats, f)

    def load_norm_stats(self, filename: str) -> dict:
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            stats = json.load(f)
        return stats
    
    def load_constant_data(self, folder: str, prefix: str, norm_stats_static: Optional[dict] = None):
        stats = norm_stats_static if norm_stats_static is not None else {}

        def standardize(data: np.ndarray, key: str) -> np.ndarray:
            """Z-score standardization"""
            if key in stats:
                mean_val = np.array(stats[key]["mean"])
                std_val = np.array(stats[key]["std"])
            else:
                mean_val = np.mean(data, axis=0)
                std_val = np.std(data, axis=0)
                stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}

            epsilon = 1e-8
            return (data - mean_val) / (std_val + epsilon)

        # Load each file using the given prefix.
        xy_path = os.path.join(folder, f"{prefix}_XY.txt")
        ca_path = os.path.join(folder, f"{prefix}_CA.txt")
        ce_path = os.path.join(folder, f"{prefix}_CE.txt")
        cs_path = os.path.join(folder, f"{prefix}_CS.txt")
        aspect_path = os.path.join(folder, f"{prefix}_A.txt")
        curvature_path = os.path.join(folder, f"{prefix}_CU.txt")
        manning_path = os.path.join(folder, f"{prefix}_N.txt")
        flow_accum_path = os.path.join(folder, f"{prefix}_FA.txt")
        infiltration_path = os.path.join(folder, f"{prefix}_IP.txt")

        xy_coords = np.loadtxt(xy_path, delimiter='\t')
        xy_coords = standardize(xy_coords, "xy_coords")
        area_denorm = np.loadtxt(ca_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        area = standardize(area_denorm, "area")
        elevation = np.loadtxt(ce_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        elevation = standardize(elevation, "elevation")
        slope = np.loadtxt(cs_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        slope = standardize(slope, "slope")
        aspect = np.loadtxt(aspect_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        aspect = standardize(aspect, "aspect")
        curvature = np.loadtxt(curvature_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        curvature = standardize(curvature, "curvature")
        manning = np.loadtxt(manning_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        manning = standardize(manning, "manning")
        flow_accum = np.loadtxt(flow_accum_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        flow_accum = standardize(flow_accum, "flow_accum")
        infiltration = np.loadtxt(infiltration_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        infiltration = standardize(infiltration, "infiltration")
        return (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
                manning, flow_accum, infiltration, stats)
    
    def create_edge_features(self, xy_coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        row, col = edge_index
        relative_coords = xy_coords[row] - xy_coords[col]
        distance = np.linalg.norm(relative_coords, axis=1)
        epsilon = 1e-8
        # Normalize relative coordinates and distance.
        relative_coords = (relative_coords - np.mean(relative_coords, axis=0)) / (np.std(relative_coords, axis=0) + epsilon)
        distance = (distance - np.mean(distance)) / (np.std(distance) + epsilon)
        return np.hstack([
            # relative_coords,
            distance[:, None],
        ])

    def load_dynamic_data(self, folder: str, hydrograph_id: str, prefix: str,
                          num_points: int, interval: int = 1, skip: int = 72):
        wd_path = os.path.join(folder, f"{prefix}_WD_{hydrograph_id}.txt")
        inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
        volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")
        precipitation_path = os.path.join(folder, f"{prefix}_Pr_{hydrograph_id}.txt")
        water_depth = np.loadtxt(wd_path, delimiter='\t')[skip::interval, :num_points]
        inflow_hydrograph = np.loadtxt(inflow_path, delimiter='\t')[skip::interval, 1]
        volume = np.loadtxt(volume_path, delimiter='\t')[skip::interval, :num_points]
        precipitation = np.loadtxt(precipitation_path, delimiter='\t')[skip::interval]
        # Limit data until 25 time steps after the peak inflow.
        peak_time_idx = np.argmax(inflow_hydrograph)
        water_depth = water_depth[:peak_time_idx + 25]
        volume = volume[:peak_time_idx + 25]
        precipitation = precipitation[:peak_time_idx + 25] * 2.7778e-7  # Unit conversion
        inflow_hydrograph = inflow_hydrograph[:peak_time_idx + 25]
        return water_depth, inflow_hydrograph, volume, precipitation
    
    def create_node_features(self, xy_coords: np.ndarray, area: np.ndarray, elevation: np.ndarray,
                             slope: np.ndarray, aspect: np.ndarray, curvature: np.ndarray,
                             manning: np.ndarray, flow_accum: np.ndarray, infiltration: np.ndarray,
                             water_depth: np.ndarray, volume: np.ndarray,
                             precipitation_data: np.ndarray, time_step: int, n_time_steps: int,
                             inflow_hydrograph: np.ndarray) -> Tuple[np.ndarray, float, float]:
        num_nodes = xy_coords.shape[0]
        # Create static copies of inflow and precipitation for each node.
        flow_hydrograph_current_step = np.full((num_nodes, 1), inflow_hydrograph[time_step])
        precip_current_step = np.full((num_nodes, 1), precipitation_data[time_step])
        # Concatenate all features horizontally.
        features = np.hstack([
            # xy_coords,
            area,
            elevation,
            slope,
            # aspect,
            # curvature,
            manning,
            # flow_accum,
            # infiltration,
            flow_hydrograph_current_step,
            precip_current_step,
            # water_depth.T,
            volume.T
        ])
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        future_precip = precipitation_data[time_step + n_time_steps]
        return features, future_inflow, future_precip

    @staticmethod
    def normalize(data: np.ndarray, mean: Union[float, list, np.ndarray],
                  std: Union[float, list, np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return (data - mean) / (std + epsilon)

    @staticmethod
    def denormalize(data: np.ndarray, mean: Union[float, list, np.ndarray],
                    std: Union[float, list, np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return data * (std + epsilon) + mean
