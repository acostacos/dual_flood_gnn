import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
