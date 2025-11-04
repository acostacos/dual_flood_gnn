import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
import pandas as pd

from data import FloodEventDataset
from data.boundary_condition import BoundaryCondition
from typing import Literal, List

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
