import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
import pandas as pd

from data import FloodEventDataset
from data.boundary_condition import BoundaryCondition
from typing import Literal, List, Tuple

def get_trimmed_cmap(cmap_name, start=0.0, end=1.0, n_colors=256):
    # Validate input parameters
    if not (0.0 <= start <= 1.0):
        raise ValueError(f"start must be in the range [0, 1], got {start}")
    if not (0.0 <= end <= 1.0):
        raise ValueError(f"end must be in the range [0, 1], got {end}")
    if start >= end:
        raise ValueError(f"start ({start}) must be less than end_pct ({end})")

    # Get the original colormap
    if isinstance(cmap_name, str):
        original_cmap = plt.get_cmap(cmap_name)
    else:
        original_cmap = cmap_name

    # Sample the colormap at the specified percentage range
    colors = original_cmap(np.linspace(start, end, n_colors))

    # Create a new colormap from the sampled colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'{original_cmap.name}_{int(start*100)}to{int(end*100)}pct',
        colors,
    )
    
    return new_cmap

def get_node_df(config: dict, run_id: str, mode: Literal['train', 'test'], no_ghost: bool = True) -> gpd.GeoDataFrame:
    '''Get the node GeoDataFrame, optionally removing ghost nodes based on boundary conditions.'''
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    nodes_shp_file = dataset_parameters['nodes_shp_file']
    nodes_shp_path = os.path.join(root_dir, 'raw', nodes_shp_file)
    node_df = gpd.read_file(nodes_shp_path)

    if no_ghost:
        summary_file_key = 'training' if mode == 'train' else 'testing'
        dataset_summary_file = dataset_parameters[summary_file_key]['dataset_summary_file']
        dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        summary_df = summary_df[summary_df['Run_ID'] == run_id]
        hec_ras_file = summary_df['HECRAS_Filepath'].values[0]

        inflow_boundary_nodes = dataset_parameters['inflow_boundary_nodes']
        outflow_boundary_nodes = dataset_parameters['outflow_boundary_nodes']

        bc = BoundaryCondition(root_dir=root_dir,
                               hec_ras_file=hec_ras_file,
                               inflow_boundary_nodes=inflow_boundary_nodes,
                               outflow_boundary_nodes=outflow_boundary_nodes,
                               saved_npz_file=FloodEventDataset.BOUNDARY_CONDITION_NPZ_FILE)
        node_df = node_df[~node_df['CC_index'].isin(bc.ghost_nodes)]

    return node_df

def get_edge_df(config: dict, run_id: str, mode: Literal['train', 'test'], no_ghost: bool = True) -> gpd.GeoDataFrame:
    '''Get the edge GeoDataFrame, optionally removing ghost edges based on boundary conditions.'''
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    edges_shp_file = dataset_parameters['edges_shp_file']
    edges_shp_path = os.path.join(root_dir, 'raw', edges_shp_file)
    link_df = gpd.read_file(edges_shp_path)

    if no_ghost:
        summary_file_key = 'training' if mode == 'train' else 'testing'
        dataset_summary_file = dataset_parameters[summary_file_key]['dataset_summary_file']
        dataset_summary_path = os.path.join(root_dir, 'raw', dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        summary_df = summary_df[summary_df['Run_ID'] == run_id]
        hec_ras_file = summary_df['HECRAS_Filepath'].values[0]

        inflow_boundary_nodes = dataset_parameters['inflow_boundary_nodes']
        outflow_boundary_nodes = dataset_parameters['outflow_boundary_nodes']

        bc = BoundaryCondition(root_dir=root_dir,
                               hec_ras_file=hec_ras_file,
                               inflow_boundary_nodes=inflow_boundary_nodes,
                               outflow_boundary_nodes=outflow_boundary_nodes,
                               saved_npz_file=FloodEventDataset.BOUNDARY_CONDITION_NPZ_FILE)
        is_ghost_edge = link_df['from_node'].isin(bc.ghost_nodes) | link_df['to_node'].isin(bc.ghost_nodes)
        boundary_nodes = np.concat([np.array(inflow_boundary_nodes), np.array(outflow_boundary_nodes)])
        is_boundary_edge = link_df['from_node'].isin(boundary_nodes) | link_df['to_node'].isin(boundary_nodes)
        link_df = pd.concat([link_df[~is_ghost_edge], link_df[is_ghost_edge & is_boundary_edge]], ignore_index=True)

        assert np.all(link_df['from_node'][bc.inflow_edges_mask].isin(inflow_boundary_nodes) | link_df['to_node'][bc.inflow_edges_mask].isin(inflow_boundary_nodes)), "Inflow of link DataFrame does not match the inflow edges mask"
        assert np.all(link_df['from_node'][bc.outflow_edges_mask].isin(outflow_boundary_nodes) | link_df['to_node'][bc.outflow_edges_mask].isin(outflow_boundary_nodes)), "Outflow of link DataFrame does not match the outflow edges mask"

    return link_df

def plot_cell_map_w_highlight(gpdf: gpd.GeoDataFrame,
                              title: str,
                              highlight_idxs: List[int]=None,
                              color_list: List[str]=None,
                              background_color: str='black',
                              legend: bool=False):
    default_marker_size = 1
    default_linewidth = 0.3
    shared_plot_kwargs = {
        'linewidth': default_linewidth,
        'markersize': default_marker_size,
    }

    if highlight_idxs is not None and len(highlight_idxs) > 0:
        # Convert colors to RGBA
        background_color_rgba = mcolors.to_rgba(background_color)
        color_list_rgba = []
        if color_list is not None:
            for color in color_list:
                if type(color) == tuple:
                    color_list_rgba.append(color)
                elif type(color) == str:
                    rgba = mcolors.to_rgba(color)
                    color_list_rgba.append(rgba)
                else:
                    raise ValueError(f"Invalid color type: {type(color)}")
        else:
            cmap = plt.get_cmap('viridis') # Default
            for i in range(len(highlight_idxs)):
                color_list_rgba.append(cmap(i / len(highlight_idxs)))

        # Create node colors and size list
        node_colors = [background_color_rgba] * len(gpdf)
        node_size = [default_marker_size] * len(gpdf)
        edge_size = [default_linewidth] * len(gpdf)
        for idx, color in zip(highlight_idxs, color_list_rgba):
            node_colors[idx] = color
            node_size[idx] = default_marker_size + 30
            edge_size[idx] = default_linewidth + 1

        shared_plot_kwargs.update({
            'markersize': node_size,
            'linewidth': edge_size,
            'color': node_colors,
            'legend': legend,
        })

    _, ax = plt.subplots()
    gpdf.plot(ax=ax, **shared_plot_kwargs)
    ax.axis('off')

    legend_handles = [mpatches.Patch(color=color_list_rgba[i], label=highlight_idxs[i]) 
                  for i in range(len(highlight_idxs))]
    ax.legend(handles=legend_handles, loc='lower right', fontsize='x-small')
    plt.title(title)
    plt.show()

def plot_loss_components(
    path: str, 
    loss_components: List[str],
    title_prefix: str,
    start_epoch: int = 0, 
    model_label: str = '', 
    log_scale: bool = False,
):
    data = np.load(path, allow_pickle=True)
    for component in loss_components:
        if component not in data:
            print(f"Component '{component}' not found in {path}. Skipping.")
            continue

        component_data = data[component][start_epoch:]
        label = component.replace('_', ' ').title()
        plt.plot(component_data, label=label)

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Log Loss')
        plt.title(f'{model_label} {title_prefix} Over Epochs (Log Scale)')
    else:
        plt.ylabel('Loss')
        plt.title(f'{model_label} {title_prefix} Over Epochs')
    
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_individual_training_loss_ratio(path: str, start_epoch: int = 0, model_label: str = None):
    loss_ratios = ['edge_scaled_loss_ratios', 'global_scaled_loss_ratios', 'local_scaled_loss_ratios']
    for ratio in loss_ratios:
        data = np.load(path, allow_pickle=True)
        if ratio not in data:
            print(f"Ratio '{ratio}' not found in {path}. Skipping.")
            continue

        train_loss_ratio = data[ratio]
        train_loss_ratio = train_loss_ratio[start_epoch:]
        label = ratio.replace('_', ' ').title()
        plt.plot(train_loss_ratio, label=label)

    plt.plot(np.ones_like(train_loss_ratio), linestyle='--', color='red', label='Target Ratio (1.0)')
    plt.title(f'{model_label if model_label is not None else ''} Training Loss Ratio Over Epochs')
    plt.xticks(np.arange(len(train_loss_ratio), step=5))
    plt.ylabel('Ratio')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def get_x_ticks_from_timestamps(metric_paths: list[str]) -> Tuple[list[str], str]:
    '''For getting the x ticks and labels in terms of time (hours) from the timestamps in the metric files.'''
    delta_t = None
    len_timestamps = None
    for path in metric_paths:
        data = np.load(path, allow_pickle=True)
        if 'timestamps' not in data or data['timestamps'] is None or len(data['timestamps']) == 0:
            continue

        timestamps = data['timestamps']
        assert len(timestamps) > 1, f"Expected more than one timestamp in {path}, found {len(timestamps)}."
        path_delta_t = timestamps[1] - timestamps[0]
        if delta_t is None:
            delta_t = path_delta_t
        assert delta_t == path_delta_t, f"Timestep interval mismatch for {path}: {delta_t} vs {path_delta_t}"
        if len_timestamps is None:
            len_timestamps = len(timestamps)
        assert len_timestamps == len(timestamps), f"Number of timestamps mismatch for {path}: {len_timestamps} vs {len(timestamps)}"
    
    if delta_t is None:
        x_ticks = np.arange(0, data['pred'].shape[0], step=10)
        return x_ticks, x_ticks, 'Timestep'

    TIME_INTERVAL_IN_HOURS = 12
    delta_t = delta_t.item()
    delta_t_in_hours = delta_t.total_seconds() / 3600
    step = TIME_INTERVAL_IN_HOURS / delta_t_in_hours
    x_ticks = np.concat([np.arange(0, len(timestamps), step=step), np.array([len(timestamps)-1])], axis=0)
    x_tick_labels = x_ticks * delta_t_in_hours
    return x_ticks, x_tick_labels, f'Time (h)'

def plot_node_and_edge_values_for_all_models(metric_paths: list[str],
                                             node_idx: int,
                                             edge_index: np.ndarray,
                                             ground_truths: Tuple[np.ndarray, np.ndarray] = (None, None),
                                             labels: list[str] = None):
    '''For plotting water volume and flow for a specific node across multiple models.'''
    vol_ground_truth, flow_ground_truth = ground_truths

    metrics = [np.load(path, allow_pickle=True) for path in metric_paths]
    if labels is None:
        labels = [path.split('/')[-1].split('_')[0] for path in metric_paths]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    x_ticks, x_ticks_labels, xlabel = get_x_ticks_from_timestamps(metric_paths)
    handles = []

    # Water Volume
    ax[0].text(1, 0.3, 'Water Volume', transform=ax[0].transAxes, rotation=270, fontsize=12)
    ax[0].set_ylabel('Water Volume (m³)')
    ax[0].set_xticks(ticks=x_ticks, labels=x_ticks_labels)

    if vol_ground_truth is not None:
       gt_line, = ax[0].plot(vol_ground_truth[:, node_idx], label='Ground Truth', color='black')
       handles.append(gt_line)

    data_shape = None
    for i, data in enumerate(metrics):
        pred = data['pred']

        if data_shape is None:
            data_shape = pred.shape
        assert pred.shape == data_shape, f"Data shape mismatch for path {i}: {pred.shape} vs {data_shape}"

        line, = ax[0].plot(pred[:, node_idx], label=labels[i])
        handles.append(line)

    connected_edges = np.nonzero(np.any(edge_index == node_idx, axis=0))[0]
    inflow_edges = connected_edges[edge_index[:, connected_edges][1] == node_idx]
    outflow_edges = connected_edges[edge_index[:, connected_edges][0] == node_idx]

    # Inflow Water Flow
    def get_inflow(flow: np.ndarray) -> np.ndarray:
        from_inflow_edges = flow[:, inflow_edges]
        from_inflow_edges[from_inflow_edges < 0] = 0 # Negative inflow = outflow
        from_outflow_edges = -flow[:, outflow_edges] # Negative outflow = inflow
        from_outflow_edges[from_outflow_edges < 0] = 0
        return from_inflow_edges.sum(axis=1) + from_outflow_edges.sum(axis=1) 

    ax[1].text(1, 0.3, 'Water Inflow', transform=ax[1].transAxes, rotation=270, fontsize=12)
    ax[1].set_ylabel('Water Flow (m³/s)')
    ax[1].set_xticks(ticks=x_ticks, labels=x_ticks_labels)

    if flow_ground_truth is not None:
        ax[1].plot(get_inflow(flow_ground_truth), label='Ground Truth', color='black')

    data_shape = None
    for i, data in enumerate(metrics):
        edge_pred = data['edge_pred']

        if data_shape is None:
            data_shape = edge_pred.shape
        assert edge_pred.shape == data_shape, f"Data shape mismatch for path {i}: {edge_pred.shape} vs {data_shape}"

        ax[1].plot(get_inflow(edge_pred), label=labels[i])

    # Outflow Water Flow
    def get_outflow(flow: np.ndarray) -> np.ndarray:
        from_outflow_edges = flow[:, outflow_edges]
        from_outflow_edges[from_outflow_edges < 0] = 0 # Negative outflow = inflow
        from_inflow_edges = -flow[:, inflow_edges] # Negative inflow = outflow
        from_inflow_edges[from_inflow_edges < 0] = 0
        return from_outflow_edges.sum(axis=1) + from_inflow_edges.sum(axis=1)

    ax[2].text(1, 0.3, 'Water Outflow', transform=ax[2].transAxes, rotation=270, fontsize=12)
    ax[2].set_ylabel('Water Flow (m³/s)')
    ax[2].set_xlabel(xlabel)
    ax[2].set_xticks(ticks=x_ticks, labels=x_ticks_labels)

    if flow_ground_truth is not None:
        ax[2].plot(get_outflow(flow_ground_truth), label='Ground Truth', color='black')

    data_shape = None
    for i, data in enumerate(metrics):
        edge_pred = data['edge_pred']

        if data_shape is None:
            data_shape = edge_pred.shape
        assert edge_pred.shape == data_shape, f"Data shape mismatch for path {i}: {edge_pred.shape} vs {data_shape}"

        ax[2].plot(get_outflow(edge_pred), label=labels[i])

    fig.suptitle(f'Metrics Over Time for Node {node_idx}')
    fig.legend(handles=handles, labels=['Ground Truth', *labels], loc='upper right')
    fig.subplots_adjust(left=0.15)
    fig.tight_layout()

def map_centers_to_polygons(polygons_gdf, centers_gdf):
    assert len(polygons_gdf) == len(centers_gdf), "Length of mesh cells DataFrame and cell centers DataFrame must be the same. Double check if you have removed ghost nodes from the cell centers."

    # Create a copy of the centers dataframe to preserve original data
    result_gdf = centers_gdf.copy()

    # Store original center coordinates for reference
    result_gdf['center_x'] = centers_gdf.geometry.x
    result_gdf['center_y'] = centers_gdf.geometry.y

    # Create spatial index for efficient querying
    spatial_index = polygons_gdf.sindex

    # Initialize list to store matched polygons
    matched_polygons = []
    matched_polygon_indices = []

    # For each center point, find the containing polygon
    for _, center in centers_gdf.iterrows():
        center_point = center.geometry

        # Get candidate polygons using spatial index
        possible_matches_idx = list(spatial_index.intersection(center_point.bounds))
        possible_matches = polygons_gdf.iloc[possible_matches_idx]

        # Find which polygon actually contains the point
        match_found = False
        for poly_idx, poly_row in enumerate(possible_matches):
            if poly_row.contains(center_point):
                matched_polygons.append(poly_row)
                matched_polygon_indices.append(poly_idx)
                match_found = True
                break

        # If no match found, append None
        if not match_found:
            matched_polygons.append(None)
            matched_polygon_indices.append(None)

    # Replace geometry with matched polygons
    result_gdf['geometry'] = matched_polygons
    result_gdf['polygon_index'] = matched_polygon_indices

    # Check for unmatched centers
    unmatched = result_gdf[result_gdf['geometry'].isna()]
    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)} cell centers could not be matched to polygons")
        print(f"Unmatched indices: {unmatched.index.tolist()}")

    return result_gdf
